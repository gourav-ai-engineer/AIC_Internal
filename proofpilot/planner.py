"""Planning: turn a goal into sub-questions, each with preferred tools and queries.

Two planners share one output format: an LLM planner (when a model is configured)
and a transparent rule-based planner that always works.
"""
from __future__ import annotations

import re
from datetime import date

from .llm import LLMClient, LLMError
from .models import SubQuestion
from .textutil import clean, key_terms, truncate
from .tools import SEARCH_TOOLS, TOOL_SPECS

INTENT_RULES = [
    ("compare", r"\b(compare|comparing|comparison|versus|vs\.?|alternatives?|which is better|trade-?offs?|pros and cons|best)\b"),
    ("latest", r"\b(latest|recent|news|today|this (?:week|month|year)|upcoming|deadlines?|current|20[2-3]\d)\b"),
    ("academic", r"\b(papers?|stud(?:y|ies)|literature|survey|benchmarks?|arxiv|scientific|peer[- ]reviewed|research papers?)\b"),
    ("howto", r"\b(how (?:to|do|can|should)|guide|tutorial|steps|set ?up|install|build|learn)\b"),
    ("decision", r"\b(should (?:i|we)|worth|decide|choose|recommend|evaluate|is it)\b"),
]
SOFTWARE = re.compile(
    r"\b(frameworks?|librar(?:y|ies)|sdks?|apis?|open[- ]source|python|javascript|typescript|packages?|github|"
    r"repos?|repositories|llms?|agents?|databases?|code|software|developer)\b", re.IGNORECASE)

MAX_SUBQUESTIONS = 5
SKEPTIC_SUFFIXES = ("limitations criticism problems", "risks drawbacks", "failures controversy")
CORE_SUFFIXES = ("official documentation", "explained guide", "report pdf", "analysis")


_LEAD_VERBS = re.compile(
    r"^(?:please\s+)?(?:compare|research|investigate|find(?: out)?|explain|summari[sz]e|analy[sz]e|evaluate|"
    r"review|explore|study|assess|tell me about|help me (?:understand|choose|decide)|what (?:is|are)|"
    r"how (?:do|does|can|to)|should (?:i|we))\s+", re.IGNORECASE)

def classify_intents(goal: str) -> list[str]:
    found = [name for name, pattern in INTENT_RULES if re.search(pattern, goal, re.IGNORECASE)]
    return found or ["general"]


def is_software(goal: str) -> bool:
    return bool(SOFTWARE.search(goal))


def extract_entities(goal: str) -> list[str]:
    """Things being compared, e.g. 'LangGraph vs CrewAI vs AutoGen' -> 3 entities."""
    text = clean(goal)
    if re.search(r"\b(?:vs\.?|versus)\b", text, re.IGNORECASE):
        parts = re.split(r"\s+(?:vs\.?|versus)\s+", text, flags=re.IGNORECASE)
        parts = [re.split(r"\s+(?:for|in|on|to|when|including)\s+|[,.;:?]", p.strip())[-1 if i == 0 else 0]
                 for i, p in enumerate(parts)]
    else:
        m = re.search(r"\b(?:compare|comparing|between)\s+(.+?)(?:\s+(?:for|in|on|including|based on|to decide|"
                      r"when|regarding)\b|[.;:?]|$)", text, re.IGNORECASE)
        parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+", m.group(1)) if m else []
    entities = [clean(re.sub(r"^(?:the|a|an)\s+", "", _LEAD_VERBS.sub("", p.strip()), flags=re.IGNORECASE))
                for p in parts]
    entities = [e for e in entities if e and len(e) <= 40 and len(e.split()) <= 4]
    return entities[:4] if len(entities) >= 2 else []


_CUT = re.compile(r"\s+(?:for|including|using|within|in the (?:next|last|past)|to (?:decide|help|choose)|so that|"
                  r"and (?:verify|check|confirm|include))\b|[,;:?]|\.(?:\s|$)", re.IGNORECASE)
_TRAILING = re.compile(r"\s+(?:in|on|of|to|and|or|with|at|by|the|a|an|can|is|are)$", re.IGNORECASE)


def _focus(goal: str) -> str:
    """Short subject phrase used to phrase sub-questions ('open-source agent frameworks')."""
    phrase = _CUT.split(_LEAD_VERBS.sub("", clean(goal)), maxsplit=1)[0]
    phrase = re.sub(r"^(?:the|a|an|which|what|how)\s+", "", phrase.strip(), flags=re.IGNORECASE)
    words = phrase.split()
    if not words:
        return " ".join(key_terms(goal, 5)) or clean(goal)[:60]
    phrase = " ".join(words[:10])
    while _TRAILING.search(phrase):
        phrase = _TRAILING.sub("", phrase)
    return phrase


def _query_base(goal: str) -> str:
    return " ".join(key_terms(goal, 6)) or clean(goal)[:80]


def heuristic_plan(goal: str, enabled: list[str] | None = None, today: date | None = None) -> list[SubQuestion]:
    enabled = enabled or list(TOOL_SPECS)
    year = (today or date.today()).year
    focus, base = _focus(goal), _query_base(goal)
    intents = classify_intents(goal)
    software = is_software(goal)
    plan: list[SubQuestion] = [
        SubQuestion("", f"What is the background and what are the key concepts behind {focus}?",
                    "Builds shared context before details.", ["wikipedia", "web_search"], [f"{base} overview"]),
        SubQuestion("", f"What do official or primary sources say about {focus}?",
                    "Primary sources beat second-hand summaries.", ["web_search"], [f"{base} official"]),
    ]
    specific: list[SubQuestion] = []
    if "compare" in intents:
        entities = extract_entities(goal)
        subject = ", ".join(entities) if entities else focus
        specific.append(SubQuestion("", f"How do {subject} compare on features, maturity, cost and trade-offs?",
                                    "The goal asks for a comparison.",
                                    ["web_search", "github"] if software else ["web_search", "news_search"],
                                    [" vs ".join(entities) if entities else f"{base} comparison"]))
    if "academic" in intents:
        specific.append(SubQuestion("", f"What does published research report about {focus}?",
                                    "The goal asks for research evidence.", ["arxiv", "web_search"], [base]))
    if "latest" in intents:
        specific.append(SubQuestion("", f"What are the most recent developments, dates or deadlines for {focus}?",
                                    "The goal is time-sensitive.", ["news_search", "web_search"],
                                    [base if str(year) in base else f"{base} {year}"]))
    if "howto" in intents:
        specific.append(SubQuestion("", f"What concrete steps or best practices apply to {focus}?",
                                    "The goal asks how to do something.", ["web_search"],
                                    [f"{base} step by step guide"]))
    if software and "compare" not in intents:
        specific.append(SubQuestion("", f"Which projects or tools for {focus} are actively maintained and adopted?",
                                    "Adoption signals indicate maturity.", ["github", "web_search"], [base]))
    if not specific:
        specific.append(SubQuestion("", f"What are real-world examples or practical implications of {focus}?",
                                    "Grounds the answer in concrete cases.", ["web_search", "news_search"],
                                    [f"{base} examples case study"]))
    skeptic = SubQuestion("", f"What are the limitations, risks or counter-evidence regarding {focus}?",
                          "Skeptic pass: actively looks for evidence against the emerging answer.",
                          ["web_search", "news_search"], [f"{base} {SKEPTIC_SUFFIXES[0]}"], role="skeptic")
    if len(specific) >= 3:  # many intents: drop the generic overview to stay within the plan size
        plan = plan[1:]
    plan = (plan + specific)[: MAX_SUBQUESTIONS - 1]
    return _finalize(plan + [skeptic], enabled)


def factcheck_plan(claim: str, enabled: list[str] | None = None) -> list[SubQuestion]:
    """Plan for checking one claim: supporting evidence, primary sources, and an active debunk search."""
    enabled = enabled or list(TOOL_SPECS)
    base = _query_base(claim)
    short = truncate(claim, 120)
    plan = [
        SubQuestion("", f"What evidence supports the claim “{short}”?",
                    "Collect the strongest evidence in favour first.", ["web_search", "news_search"], [base]),
        SubQuestion("", f"What do official or primary sources say about “{short}”?",
                    "Primary sources settle factual questions.", ["web_search", "wikipedia"], [f"{base} official"]),
    ]
    if re.search(INTENT_RULES[2][1], claim, re.IGNORECASE) or re.search(r"\b(health|study|cause|effect)", claim,
                                                                       re.IGNORECASE):
        plan.append(SubQuestion("", f"What does published research say about “{short}”?",
                                "Scientific claims need research evidence.", ["arxiv", "web_search"], [base]))
    plan.append(SubQuestion("", f"Has “{short}” been fact-checked, disputed or debunked?",
                            "Skeptic pass: actively searches for evidence that the claim is false.",
                            ["web_search", "news_search"], [f"{base} fact check", f"{base} false myth debunked"],
                            role="skeptic"))
    return _finalize(plan, enabled)


def _finalize(subquestions: list[SubQuestion], enabled: list[str]) -> list[SubQuestion]:
    searchers = [t for t in enabled if t in SEARCH_TOOLS] or ["web_search"]
    for i, sq in enumerate(subquestions, 1):
        sq.id = f"Q{i}"
        sq.tools = [t for t in dict.fromkeys(sq.tools) if t in searchers] or [searchers[0]]
        for t in ("web_search", *searchers):  # always leave a general fallback tool
            if t in searchers and t not in sq.tools:
                sq.tools.append(t)
                break
        sq.queries = [truncate(q, 120) for q in sq.queries if clean(q)] or [truncate(sq.question, 120)]
    return subquestions


PLAN_PROMPT = """You are the planning module of ProofPilot, an evidence-first research agent.
Today is {today}. Break the GOAL into 3-5 sub-questions that together answer it completely.
Exactly one sub-question must have role "skeptic": it searches for limitations, criticism or counter-evidence.
For each sub-question choose 1-3 tools (most useful first) from:
{tools}
and write 1-2 short search-engine queries (keywords, not sentences; include years when recency matters).
Return JSON only, in this shape:
{{"intent": "<one phrase>", "subquestions": [{{"question": "...", "rationale": "<why it matters>",
"role": "core|skeptic", "tools": ["web_search"], "queries": ["..."]}}]}}

GOAL: {goal}"""


FACTCHECK_NOTE = ("\nThis is a FACT-CHECK. The goal is a single claim: plan sub-questions that look for evidence FOR it, "
                  "primary sources about it, and (skeptic) evidence that it is false, outdated or misleading.")


def llm_plan(goal: str, llm: LLMClient, enabled: list[str] | None = None,
             today: date | None = None, factcheck: bool = False) -> list[SubQuestion]:
    """Ask the model for a plan; validate it strictly. Raises LLMError if unusable."""
    enabled = enabled or list(TOOL_SPECS)
    tools = "\n".join(f"- {n}: {TOOL_SPECS[n].description}" for n in enabled if n in SEARCH_TOOLS)
    prompt = PLAN_PROMPT.format(today=(today or date.today()).isoformat(), tools=tools, goal=goal)
    data = llm.complete_json(prompt + (FACTCHECK_NOTE if factcheck else ""), max_tokens=1200)
    raw = data.get("subquestions")
    if not isinstance(raw, list):
        raise LLMError("plan missing subquestions")
    plan = []
    for item in raw[:MAX_SUBQUESTIONS]:
        if not isinstance(item, dict) or not clean(str(item.get("question", ""))):
            continue
        tools_ = [t for t in item.get("tools", []) if isinstance(t, str)]
        queries = [str(q) for q in item.get("queries", []) if isinstance(q, (str, int))][:2]
        plan.append(SubQuestion("", truncate(str(item["question"]), 220), truncate(str(item.get("rationale", "")), 200),
                                tools_, queries, role="skeptic" if item.get("role") == "skeptic" else "core"))
    if len(plan) < 2:
        raise LLMError("plan has fewer than 2 usable sub-questions")
    if not any(sq.role == "skeptic" for sq in plan):
        focus = _focus(goal)
        base = _query_base(goal)
        plan = plan[: MAX_SUBQUESTIONS - 1] + [SubQuestion(
            "", f"What are the limitations, risks or counter-evidence regarding {focus}?",
            "Skeptic pass added by guardrail.", ["web_search"], [f"{base} {SKEPTIC_SUFFIXES[0]}"], role="skeptic")]
    return _finalize(plan, enabled)


def reformulate(sq: SubQuestion, goal: str, tool: str) -> str | None:
    """Next untried query for a weakly-covered sub-question, or None if ideas are exhausted."""
    base = " ".join(dict.fromkeys(key_terms(sq.question, 4) + key_terms(goal, 3)))
    suffixes = SKEPTIC_SUFFIXES if sq.role == "skeptic" else CORE_SUFFIXES
    for candidate in [*sq.queries, *(f"{base} {s}" for s in suffixes), base]:
        candidate = truncate(candidate, 120)
        if not sq.attempted(tool, candidate):
            return candidate
    return None

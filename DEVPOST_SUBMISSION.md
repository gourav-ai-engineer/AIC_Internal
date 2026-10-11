# Devpost submission — copy-paste ready

> Fill in the two links marked **TODO** before submitting. Everything below describes features that exist in the
> repository. Do not add numbers you have not measured yourself.

## Project name
ProofPilot — the research agent that doubts itself

## Tagline (≤ 200 chars)
An autonomous research agent that plans, chooses its own tools, hunts for counter-evidence, and checks every claim
against its sources before you see it.

## Links
- Source code: https://github.com/gourav-ai-engineer/AIC_Internal
- Demo video: **TODO** (YouTube / Vimeo, unlisted is fine)
- Live app (optional): **TODO** (Streamlit Community Cloud URL)

---

## Inspiration
Every student has asked an AI a research question and received a confident answer with a citation that doesn't say
what the AI claimed, or doesn't exist at all. Chatbots answer once and never check themselves. We wanted an agent
that behaves like a careful researcher: break the problem down, gather evidence from several places, look for
reasons it might be wrong, and **verify before it speaks**.

## What it does
You give ProofPilot a goal, such as *"Compare LangGraph vs CrewAI vs AutoGen for a student project"*. It then works
on its own:

1. **Recalls** related past research from long-term memory and reuses trusted sources.
2. **Plans** 3–5 sub-questions, always including a **skeptic pass** that looks for limitations and counter-evidence.
3. **Runs a decide → act → observe loop.** At every step the controller chooses one tool (web search, news,
   Wikipedia, arXiv, GitHub, or reading a full page) or decides to finish, and explains why.
4. **Adapts.** Empty results trigger reformulated queries or a different tool. A failing tool is disabled by a
   circuit breaker and work is rerouted. If the LLM is rate-limited, the agent switches to its heuristic controller
   mid-run.
5. **Reflects** on coverage: each sub-question needs evidence from 2+ independent domains and at least one fully
   read page. The agent stops early when everything is covered.
6. **Verifies.** Every finding is checked against the text of the sources it cites. Fabricated citations are
   removed, unsupported claims are marked 🔴, single-source claims are marked 🟠, and claims backed by several
   independent domains are marked 🟢.
7. **Reports** a brief with clickable citations, an evidence ledger with credibility scores, the full decision
   trace, a coverage view, an evidence map graph, and an overall **evidence strength** score. Downloads are
   available as Markdown and JSON.

## Why it is agentic
ProofPilot is a goal-driven loop, not a single prompt:
- **Planning:** goal decomposition with per-sub-question tool choices and a rationale.
- **Tool use:** 6 tools with no required API keys, chosen dynamically at each step.
- **Reasoning:** every decision records a "thought" (ReAct style), visible live in the UI.
- **Observation and adaptation:** results, failures and coverage feed the next decision. Includes reformulation,
  tool switching, circuit breakers and LLM fallback.
- **Memory:** runs persist and related runs are recalled to seed new research.
- **Autonomy with guardrails:** a hard tool budget and time limit, early stopping, a guard against finishing before
  every sub-question has been researched, and a rule that the LLM may only fetch URLs a tool actually returned.
- **Self-verification:** the agent audits its own output against the evidence.

## How we built it
- **Python** package `proofpilot/` with a clean module per responsibility: controller, planner, tools, LLM client,
  verifier, credibility, report, memory, offline mode, CLI.
- **Streamlit** dashboard that streams agent events live, plus a Graphviz evidence map.
- **Tools:** DuckDuckGo search and news (`ddgs`), the Wikipedia API, the arXiv API, the GitHub search API, and a page
  reader built on Requests and Beautiful Soup.
- **LLM (optional):** a provider-agnostic client over plain HTTP for **Gemini**, **Groq**, any OpenAI-compatible
  endpoint, or local **Ollama**, using JSON-mode prompts with strict validation.
- **Verification:** lexical grounding (sentence-level and whole-text term coverage) and clustering across
  independent domains.
- **Safety:** prompt-injection detection and removal, an SSRF guard that re-checks every redirect, and API keys sent
  in headers (never in URLs) and never stored.
- **Quality:** 59 offline unit and behaviour tests, including a headless UI test, with GitHub Actions CI on
  Python 3.10 and 3.12.

## Challenges we ran into
- **LLMs make up citations.** We stopped trusting the writer and built a verifier that re-reads the sources and
  removes or downgrades anything it can't find.
- **Free APIs fail and rate-limit.** We made failure a normal observation: circuit breakers for tools, a fallback
  controller for the LLM, and a deterministic offline mode so the demo never depends on Wi-Fi.
- **Web pages can attack agents.** Pages can contain text like "ignore previous instructions". We detect and strip
  it before the model sees it, and lower that source's credibility.
- **Knowing when to stop.** Coverage-based reflection lets the agent stop early instead of burning its budget.

## Accomplishments we're proud of
- The agent works **without any API key**, and the same loop gets smarter with an LLM.
- Every claim in the report carries a confidence level that is explained, not just asserted.
- The full trace makes the agent's behaviour auditable. Judges can see *why* each tool was chosen.

## What we learned
Agentic systems are mostly about handling the unhappy path: tools fail, models misbehave, and content is
untrusted. Making the process observable made debugging, and trust, much easier.

## What's next
- Semantic (embedding-based) verification alongside lexical grounding.
- Contradiction detection between sources.
- Scheduled "watch this topic" runs that alert you when the evidence changes.
- PDF reading and export.

## Built with
python · streamlit · requests · beautifulsoup4 · ddgs · wikipedia-api · arxiv-api · github-api · gemini · groq ·
graphviz · github-actions

## How to run
See the [README](https://github.com/gourav-ai-engineer/AIC_Internal#quick-start-windows-powershell). In short:
`pip install -r requirements.txt` then `streamlit run app.py`. No key is needed; for the best results add a free
Gemini key. **Offline demo mode** works without internet.

---

## Final checklist
- [ ] `python -m unittest discover -v` passes on your machine.
- [ ] One **live** run (offline mode off) completed with real sources, ideally with a Gemini key.
- [ ] Demo video recorded (see `DEMO_SCRIPT.md`), uploaded, and the link added above.
- [ ] No API key visible in the video, the screenshots or the repository.
- [ ] Optional: deploy to Streamlit Community Cloud and add the URL.
- [ ] Add 3–5 screenshots to the Devpost gallery: live trace, verified findings, evidence map, brief.
- [ ] Submit before **Oct 12, 2026, 9:30 AM IST**.

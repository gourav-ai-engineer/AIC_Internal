"""Fact-check mode: turn "Is it true that ...?" into an evidence-weighted verdict.

The verdict is computed from the evidence (who supports the claim, who contradicts or
debunks it, and how credible each side is), never taken on an LLM's word.
"""
from __future__ import annotations

import re

from .conflicts import DEBUNK, compare
from .models import Evidence
from .textutil import clean, content_terms, coverage, sentences
from .verify import support_score

_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:fact[- ]?check(?:ing)?(?:\s+this)?(?:\s+claim)?|verify(?:\s+(?:this|the))?\s+claim|"
    r"is\s+it\s+true\s+that|true\s+or\s+false|claim)\s*[:\-–—,]?\s*", re.IGNORECASE)

VERDICTS = {
    "Supported": "🟢", "Mostly supported": "🟢", "Weakly supported": "🟡", "Disputed": "🟠",
    "Mostly refuted": "🔴", "Refuted": "🔴", "Unverified": "⚪",
}


def is_factcheck(goal: str) -> bool:
    m = _PREFIX.match(goal or "")
    return bool(m) and len(goal[m.end():].strip()) >= 8


def claim_of(goal: str) -> str:
    """The claim being checked, without the 'Fact-check:' prefix or trailing question mark."""
    return clean(_PREFIX.sub("", goal or "", count=1)).rstrip("?").strip() or clean(goal)


def refutes(claim: str, text: str) -> str:
    """Why `text` contradicts or debunks `claim` ('' if it does not)."""
    terms = content_terms(claim)
    for s in sentences(text):
        if coverage(terms, content_terms(s)) < 0.4:
            continue
        if DEBUNK.search(s):
            return f"calls it false/misleading: “{s[:160]}”"
        hit = compare(claim, s)
        if hit:
            return f"{hit[0]}: “{s[:160]}”"
    return ""


def _weight(items: list[Evidence]) -> tuple[int, float]:
    best: dict[str, float] = {}
    for e in items:
        best[e.domain] = max(best.get(e.domain, 0.0), e.credibility)
    return len(best), sum(best.values())


def judge(claim: str, cited: list[Evidence]) -> dict:
    supporting, refuting, reasons = [], [], {}
    for e in cited:
        why = refutes(claim, e.text)
        if why:
            refuting.append(e)
            reasons[e.sid] = why
        elif support_score(claim, e.text) >= 0.6:
            supporting.append(e)
    (sd, sw), (rd, rw) = _weight(supporting), _weight(refuting)
    if sd == 0 and rd == 0:
        verdict = "Unverified"
    elif rd == 0:
        verdict = "Supported" if sd >= 2 else "Weakly supported"
    elif sd == 0:
        verdict = "Refuted" if rd >= 2 or rw >= 0.75 else "Disputed"
    elif sw >= 2 * rw:
        verdict = "Mostly supported"
    elif rw >= 2 * sw:
        verdict = "Mostly refuted"
    else:
        verdict = "Disputed"
    explanation = (f"{sd} independent domain(s) support the claim (credibility weight {sw:.2f}); "
                   f"{rd} contradict or debunk it (weight {rw:.2f}).")
    if verdict == "Unverified":
        explanation += " No source clearly addressed the claim, so ProofPilot will not guess."
    return {
        "claim": claim, "verdict": verdict, "icon": VERDICTS[verdict], "explanation": explanation,
        "supporting": [e.sid for e in supporting], "refuting": [e.sid for e in refuting],
        "refuting_reasons": reasons, "support_domains": sd, "refute_domains": rd,
    }

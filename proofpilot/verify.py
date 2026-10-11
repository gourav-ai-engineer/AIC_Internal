"""Verification: cross-source corroboration and citation grounding.

ProofPilot does not trust its own report. Every claim is checked against the text of
the sources it cites, and confidence comes from how many *independent domains*
actually support it.
"""
from __future__ import annotations

import re

from .models import Claim, Evidence
from .textutil import content_terms, coverage, entities, jaccard, sentences

GROUNDED = 0.45  # minimum share of a claim's terms a source must contain to count as support
CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Unsupported": 0}
_CITE = re.compile(r"\bS(\d{1,3})\b")


def support_score(claim: str, text: str) -> float:
    """How well `text` supports `claim`: best single sentence blended with whole-text term coverage."""
    terms = content_terms(claim)
    if not terms or not text:
        return 0.0
    names = entities(claim)
    predicate = terms - names  # what is said about the subject, e.g. {"visible", "space"}
    best, best_pred = 0.0, 0.0
    for s in sentences(text):
        st = content_terms(s)
        cov = coverage(terms, st)
        if cov > best:
            best, best_pred = cov, coverage(predicate, st)
    score = 0.65 * best + 0.35 * coverage(terms, content_terms(text))
    if names and not names & content_terms(text):  # e.g. a source about Go cannot confirm a claim about Rust
        score *= 0.5
    if names and len(predicate) >= 2 and best_pred < 0.4:  # mentions the subject but not what is claimed about it
        score *= 0.6
    return round(score, 3)


def confidence(domains: int, grounding: float, mean_credibility: float) -> str:
    if domains == 0 or grounding < GROUNDED:
        return "Unsupported"
    if domains >= 3 or (domains == 2 and mean_credibility >= 0.7):
        return "High"
    if domains == 2:
        return "Medium"
    return "Low"


def audit(text: str, cited: list[str], by_sid: dict[str, Evidence]) -> Claim:
    """Check a claim against the sources it cites, plus any other source that also supports it."""
    valid = [s for s in dict.fromkeys(cited) if s in by_sid]
    dropped = [s for s in dict.fromkeys(cited) if s not in by_sid]
    scores = {sid: support_score(text, by_sid[sid].text) for sid in valid}
    supporting = [sid for sid, sc in scores.items() if sc >= GROUNDED]
    for sid, e in by_sid.items():  # independent corroboration beyond the cited set
        if sid not in scores and support_score(text, e.text) >= 0.6:
            supporting.append(sid)
    domains = len({by_sid[s].domain for s in supporting})
    grounding = max(scores.values(), default=0.0) if valid else 0.0
    if supporting and not valid:
        grounding = max(support_score(text, by_sid[s].text) for s in supporting)
    cred = sum(by_sid[s].credibility for s in supporting) / len(supporting) if supporting else 0.0
    notes = []
    if dropped:
        notes.append(f"removed non-existent citation(s) {', '.join(dropped)}")
    weak = [s for s in valid if s not in supporting]
    if weak:
        notes.append(f"{', '.join(weak)} did not clearly contain this claim")
    if domains == 1:
        notes.append("single-source claim: needs independent confirmation")
    extra = [s for s in supporting if s not in valid]
    if extra:
        notes.append(f"also corroborated by {', '.join(extra)}")
    ordered = sorted(supporting or valid, key=lambda s: int(s[1:]))
    return Claim(text, ordered, domains, round(grounding, 2), confidence(domains, grounding, cred),
                 "; ".join(notes), list(dict.fromkeys(cited)))


def _candidate_sentences(e: Evidence, goal_terms: set[str]) -> list[tuple[float, str]]:
    out = []
    for s in sentences(e.text):
        if not 50 <= len(s) <= 320 or s.endswith("?"):
            continue
        rel = coverage(goal_terms, content_terms(s))
        if rel == 0:
            continue
        specific = 0.15 if re.search(r"\d", s) else 0.0
        out.append((rel + specific + 0.2 * e.credibility, s))
    return sorted(out, reverse=True)[:4]


def corroborate(evidence: list[Evidence], goal: str, max_claims: int = 8) -> list[Claim]:
    """Extractive claim mining: find statements that several independent sources agree on."""
    goal_terms = content_terms(goal)
    clusters: list[dict] = []
    for e in evidence:
        for value, s in _candidate_sentences(e, goal_terms):
            terms = content_terms(s)
            home = next((c for c in clusters if jaccard(terms, c["terms"]) >= 0.4), None)
            if home is None:
                clusters.append({"terms": terms, "text": s, "value": value, "sids": [e.sid]})
            elif e.sid not in home["sids"]:
                home["sids"].append(e.sid)
                if value > home["value"]:
                    home.update(text=s, value=value)
    by_sid = {e.sid: e for e in evidence}
    claims = [audit(c["text"], c["sids"], by_sid) for c in clusters]
    claims.sort(key=lambda c: (CONFIDENCE_ORDER[c.confidence], c.domains, c.grounding), reverse=True)
    unique: list[Claim] = []
    for c in claims:
        if all(jaccard(content_terms(c.text), content_terms(u.text)) < 0.5 for u in unique):
            unique.append(c)
    # keep a couple of weakly-supported statements visible, so the user sees what is NOT corroborated
    strong = [c for c in unique if c.domains >= 2][: max(1, max_claims - 2)]
    weak = [c for c in unique if c.domains < 2]
    rest = [c for c in unique if c.domains >= 2 and c not in strong]
    return (strong + weak + rest)[:max_claims]


def cited_ids(text: str) -> list[str]:
    return [f"S{n}" for n in _CITE.findall(text or "")]

"""Source credibility scoring and prompt-injection defence.

Scores are transparent heuristics, not truth judgements: they help the agent decide
what to read first and tell the user how much weight a source deserves.
"""
from __future__ import annotations

import re
from datetime import date

from .models import Evidence
from .textutil import sentences

OFFICIAL_SUFFIXES = (".gov", ".mil", ".int", ".gov.in", ".nic.in", ".gov.uk", ".gc.ca", ".gov.au", ".europa.eu")
OFFICIAL_DOMAINS = ("who.int", "un.org", "oecd.org", "worldbank.org", "nist.gov", "europa.eu")
ACADEMIC_SUFFIXES = (".edu", ".ac.in", ".ac.uk", ".edu.in", ".edu.au", ".ac.jp")
ACADEMIC_DOMAINS = (
    "arxiv.org", "nature.com", "science.org", "sciencedirect.com", "springer.com", "ieee.org", "acm.org",
    "nih.gov", "semanticscholar.org", "openreview.net", "aclanthology.org", "plos.org", "jstor.org",
    "pubmed.ncbi.nlm.nih.gov", "scholar.google.com", "researchgate.net", "mdpi.com", "wiley.com",
)
REFERENCE_DOMAINS = ("wikipedia.org", "britannica.com")
NEWS_DOMAINS = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com", "washingtonpost.com",
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com", "thehindu.com", "indianexpress.com",
    "ndtv.com", "hindustantimes.com", "timesofindia.indiatimes.com", "cnbc.com", "bloomberg.com", "ft.com",
    "economist.com", "venturebeat.com", "zdnet.com", "engadget.com", "livemint.com", "economictimes.indiatimes.com",
)
CODE_DOMAINS = ("github.com", "gitlab.com", "pypi.org", "npmjs.com", "huggingface.co", "crates.io")
COMMUNITY_DOMAINS = (
    "reddit.com", "quora.com", "medium.com", "stackexchange.com", "stackoverflow.com", "news.ycombinator.com",
    "dev.to", "substack.com", "x.com", "twitter.com", "facebook.com", "linkedin.com", "youtube.com",
    "pinterest.com", "tiktok.com", "blogspot.com", "wordpress.com",
)
DOCS_PREFIXES = ("docs.", "developer.", "developers.", "learn.", "api.", "devdocs.")
DOCS_SUFFIXES = (".readthedocs.io", ".gitbook.io")

TOOL_TYPES = {
    "arxiv": ("academic preprint", 0.8),
    "wikipedia": ("reference", 0.7),
    "github": ("code repository", 0.7),
    "news_search": ("news", 0.65),
}

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts|rules)",
        r"disregard (?:all |any |the )?(?:previous|prior|above|earlier)",
        r"forget (?:all |your )?(?:previous |prior )?instructions",
        r"you are now (?:a|an|in|the)\b",
        r"(?:reveal|print|show|output) (?:your|the) (?:system|hidden) prompt",
        r"new (?:system )?instructions\s*:",
        r"<\s*/?\s*(?:system|assistant|instructions?)\s*>",
        r"do not (?:tell|inform) the user",
        r"(?:assistant|ai|llm|model)s? (?:must|should) (?:now )?(?:say|respond|answer|output)",
    )
]


def _matches(domain: str, names: tuple[str, ...]) -> bool:
    return any(domain == n or domain.endswith("." + n) for n in names)


def classify_source(e: Evidence) -> tuple[str, float]:
    """Return (source type, base credibility 0..1) from the domain and the tool that found it."""
    d = e.domain
    if d.endswith(OFFICIAL_SUFFIXES) or _matches(d, OFFICIAL_DOMAINS):
        return "official / government", 0.9
    if d.endswith(ACADEMIC_SUFFIXES):
        return "academic institution", 0.85
    if _matches(d, ACADEMIC_DOMAINS):
        return "academic publication", 0.85
    if d.startswith(DOCS_PREFIXES) or d.endswith(DOCS_SUFFIXES):
        return "official documentation", 0.8
    if _matches(d, REFERENCE_DOMAINS):
        return "reference", 0.7
    if _matches(d, NEWS_DOMAINS):
        return "established news", 0.72
    if _matches(d, CODE_DOMAINS):
        return "code repository", 0.7
    if _matches(d, COMMUNITY_DOMAINS):
        return "community / forum", 0.4
    if e.tool in TOOL_TYPES:
        return TOOL_TYPES[e.tool]
    return "general web", 0.55


def _year(published: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", published or "")
    return int(m.group(0)) if m else None


def score(e: Evidence, today: date | None = None) -> Evidence:
    """Attach source type, a credibility score and its label to the evidence (in place)."""
    today = today or date.today()
    source_type, value = classify_source(e)
    if e.read:
        value += 0.05
    year = _year(e.published)
    if year is not None:
        age = today.year - year
        if age <= 1:
            value += 0.05
        elif age >= 5:
            value -= 0.1
    if any(f.startswith("injection") for f in e.flags):
        value -= 0.3
    if e.status.startswith(("fetch-error", "blocked")):
        value -= 0.05
    e.source_type = source_type
    e.credibility = round(max(0.0, min(1.0, value)), 2)
    e.credibility_label = label(e.credibility)
    return e


def label(value: float) -> str:
    return "High" if value >= 0.75 else "Medium" if value >= 0.5 else "Low"


def scan_injection(text: str) -> list[str]:
    """Names of prompt-injection patterns present in untrusted text."""
    return [f"injection:{p.pattern[:40]}" for p in INJECTION_PATTERNS if p.search(text or "")]


def neutralize(text: str) -> tuple[str, int]:
    """Drop sentences that look like instructions aimed at an AI; return (clean text, removed count)."""
    kept, removed = [], 0
    for s in sentences(text):
        if any(p.search(s) for p in INJECTION_PATTERNS):
            removed += 1
        else:
            kept.append(s)
    return " ".join(kept), removed

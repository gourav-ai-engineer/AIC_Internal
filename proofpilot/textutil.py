"""Dependency-free text helpers used for planning, ranking and verification."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more most my no nor not now
    of off on once only or other our out over own same she should so some such than that the their them
    then there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours i'm it's let's get got want like may might must per
    including include includes using use used make need needs find show shows research researching
    investigate compare comparing comparison latest recent official sources source evidence verify
    information details detail best good based within next days give tell explain help me please
    """.split()
)

SHORT_TERMS = frozenset({"ai", "ml", "go", "ui", "ux", "vr", "3d", "5g", "os", "qa"})

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*[a-z0-9+#]|[a-z0-9]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_SPACE = re.compile(r"\s+")


def clean(text: str | None) -> str:
    """Collapse whitespace and strip."""
    return _SPACE.sub(" ", text or "").strip()


def tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _meaningful(word: str) -> bool:
    return (len(word) > 2 or word in SHORT_TERMS) and word not in STOPWORDS


def stem(word: str) -> str:
    """Very light plural folding so 'teams' matches 'team' and 'studies' matches 'study'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_terms(text: str) -> set[str]:
    """Meaningful lower-case terms (no stopwords, at least 3 characters), plural-folded for matching."""
    return {stem(w) for w in tokens(text) if _meaningful(w)}


_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9+#\-]*[a-zA-Z0-9+#]|\b\d+(?:\.\d+)+\b")


def entities(text: str) -> set[str]:
    """Proper-noun-like terms (capitalised words, version numbers) that identify what a claim is about."""
    found = set()
    for m in _ENTITY.finditer(text or ""):
        w = m.group(0).lower()
        if w not in STOPWORDS and len(w) > 1 and w not in _COMMON_CAPS:
            found.add(stem(w))
    return found


_COMMON_CAPS = frozenset("""many most some several independent official recent new first one two three according
however today yesterday researchers experts students users developers studies surveys""".split())


def key_terms(text: str, limit: int = 6) -> list[str]:
    """Content terms in their original order, de-duplicated."""
    seen: list[str] = []
    for w in tokens(text):
        if _meaningful(w) and w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def sentences(text: str) -> list[str]:
    return [clean(s) for s in _SENTENCE_SPLIT.split(text or "") if clean(s)]


def coverage(needle: set[str], haystack: set[str]) -> float:
    """Share of `needle` terms found in `haystack` (0..1)."""
    return len(needle & haystack) / len(needle) if needle else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def truncate(text: str, limit: int) -> str:
    text = clean(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"

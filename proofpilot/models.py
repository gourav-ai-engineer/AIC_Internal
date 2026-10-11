"""Core data structures shared across ProofPilot modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_", "ref_src")


def normalize_url(url: str) -> str:
    """Canonical form used to de-duplicate sources (drops fragments and tracking params)."""
    p = urlparse((url or "").strip())
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(_TRACKING_PARAMS)]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", urlencode(query), ""))


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


@dataclass
class Evidence:
    """One source the agent found, plus everything it learned about it."""

    title: str
    url: str
    snippet: str
    tool: str = "web_search"
    query: str = ""
    subquestion: str = ""
    page_text: str = ""
    status: str = "search-result"
    published: str = ""
    credibility: float = 0.0
    credibility_label: str = ""
    source_type: str = ""
    flags: list[str] = field(default_factory=list)
    sid: str = ""

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    @property
    def text(self) -> str:
        """Best available text: full page if read, otherwise the search snippet."""
        return self.page_text or self.snippet

    @property
    def read(self) -> bool:
        return bool(self.page_text)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["domain"] = self.domain
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class SubQuestion:
    """A piece of the user's goal that the agent researches separately."""

    id: str
    question: str
    rationale: str = ""
    tools: list[str] = field(default_factory=lambda: ["web_search"])
    queries: list[str] = field(default_factory=list)
    role: str = "core"
    tried: list[str] = field(default_factory=list)

    def attempted(self, tool: str, query: str) -> bool:
        return f"{tool}::{query.strip().lower()}" in self.tried

    def mark(self, tool: str, query: str) -> None:
        self.tried.append(f"{tool}::{query.strip().lower()}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    """A statement in the report together with the verification result."""

    text: str
    sources: list[str]
    domains: int = 0
    grounding: float = 0.0
    confidence: str = "Low"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    """What the controller decided to do next, and why."""

    action: str
    thought: str
    subquestion: str = ""
    query: str = ""
    url: str = ""
    by: str = "heuristic"

"""Tool layer: every external capability the agent can call.

All tools are key-less (free) so the project runs for anyone. Each tool returns
`Evidence` objects; failures raise `ToolError` so the controller can adapt.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import offline
from .models import Evidence, normalize_url
from .textutil import clean, key_terms, truncate

USER_AGENT = "ProofPilot/2.0 (student research agent; +https://github.com/gourav-ai-engineer/AIC_Internal)"
PAGE_CHARS = 8000
MAX_PAGE_BYTES = 2_000_000
TIMEOUT = (5, 12)


class ToolError(Exception):
    """A tool call failed in a way the agent should observe and adapt to."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    kind: str  # "search" or "read"


TOOL_SPECS: dict[str, ToolSpec] = {
    "web_search": ToolSpec("web_search", "General web search. Best for official pages, documentation and guides.", "search"),
    "news_search": ToolSpec("news_search", "Recent news with publication dates. Best for latest events, announcements and deadlines.", "search"),
    "wikipedia": ToolSpec("wikipedia", "Encyclopedia summaries. Best for definitions, background and overviews.", "search"),
    "arxiv": ToolSpec("arxiv", "Academic preprints. Best for research papers, methods and benchmarks.", "search"),
    "github": ToolSpec("github", "GitHub repositories with stars and activity. Best for software, libraries and frameworks.", "search"),
    "fetch_page": ToolSpec("fetch_page", "Read the full text of a URL that a search tool already returned.", "read"),
}
SEARCH_TOOLS = tuple(name for name, spec in TOOL_SPECS.items() if spec.kind == "search")


def _get(url: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r
    except requests.RequestException as ex:
        raise ToolError(f"{type(ex).__name__}") from ex


# --------------------------------------------------------------------------- search tools


def web_search(query: str, max_results: int = 6) -> list[Evidence]:
    from ddgs import DDGS  # imported lazily so offline runs and tests never need the network

    try:
        with DDGS(timeout=10) as ddgs:
            rows = ddgs.text(query, max_results=max_results) or []
    except Exception as ex:  # ddgs raises its own exception hierarchy
        raise ToolError(type(ex).__name__) from ex
    out = []
    for r in rows:
        url = str(r.get("href") or r.get("url") or "")
        if url.startswith(("http://", "https://")):
            out.append(Evidence(truncate(str(r.get("title") or "Untitled"), 200), url,
                                truncate(str(r.get("body") or ""), 900), tool="web_search", query=query))
    return out


def news_search(query: str, max_results: int = 5) -> list[Evidence]:
    from ddgs import DDGS

    try:
        with DDGS(timeout=10) as ddgs:
            rows = ddgs.news(query, max_results=max_results) or []
    except Exception as ex:
        raise ToolError(type(ex).__name__) from ex
    out = []
    for r in rows:
        url = str(r.get("url") or r.get("href") or "")
        if url.startswith(("http://", "https://")):
            outlet = str(r.get("source") or "")
            title = truncate(str(r.get("title") or "Untitled"), 200)
            out.append(Evidence(f"{title} ({outlet})" if outlet else title, url,
                                truncate(str(r.get("body") or ""), 900), tool="news_search", query=query,
                                published=str(r.get("date") or "")[:10]))
    return out


def wikipedia_search(query: str, max_results: int = 3) -> list[Evidence]:
    r = _get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": max_results,
        "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": max_results, "inprop": "url",
        "redirects": 1,
    })
    try:
        pages = sorted(r.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    except ValueError as ex:
        raise ToolError("InvalidJSON") from ex
    out = []
    for p in pages:
        extract, url = clean(p.get("extract", "")), p.get("fullurl", "")
        if extract and url:
            out.append(Evidence(f"{p.get('title', 'Wikipedia')} — Wikipedia", url, truncate(extract, 600),
                                tool="wikipedia", query=query, page_text=extract[:PAGE_CHARS], status="fetched:api"))
    return out


_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_search(query: str, max_results: int = 4) -> list[Evidence]:
    terms = key_terms(query, 5) or [query]
    out: list[Evidence] = []
    for joiner in (" AND ", " OR "):  # strict first, then relaxed if nothing matches
        r = _get("https://export.arxiv.org/api/query", params={
            "search_query": joiner.join(f"all:{t}" for t in terms), "start": 0,
            "max_results": max_results, "sortBy": "relevance",
        })
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as ex:
            raise ToolError("InvalidXML") from ex
        for entry in root.findall("a:entry", _ATOM):
            title = clean(entry.findtext("a:title", "", _ATOM))
            url = clean(entry.findtext("a:id", "", _ATOM)).replace("http://", "https://")
            summary = clean(entry.findtext("a:summary", "", _ATOM))
            authors = [clean(a.findtext("a:name", "", _ATOM)) for a in entry.findall("a:author", _ATOM)]
            if title and url:
                byline = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
                out.append(Evidence(f"{title} (arXiv)", url, truncate(summary, 600), tool="arxiv", query=query,
                                    page_text=f"Authors: {byline}. Abstract: {summary}"[:PAGE_CHARS],
                                    status="fetched:api", published=entry.findtext("a:published", "", _ATOM)[:10]))
        if out:
            break
    return out


def github_search(query: str, max_results: int = 5) -> list[Evidence]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    terms = [t for t in key_terms(query, 4) if t not in ("vs", "versus")] or [query]
    items: list = []
    for q in (" ".join(terms[:3]), " OR ".join(terms[:3])):  # all terms first, then any term
        r = _get("https://api.github.com/search/repositories", params={"q": q, "per_page": max_results},
                 headers=headers)
        try:
            items = r.json().get("items", [])
        except ValueError as ex:
            raise ToolError("InvalidJSON") from ex
        if items or len(terms) == 1:
            break
    out = []
    for it in items:
        lic = (it.get("license") or {}).get("spdx_id") or "unknown"
        facts = (f"{it.get('description') or 'No description.'} — {it.get('stargazers_count', 0):,} stars, "
                 f"{it.get('forks_count', 0):,} forks, language {it.get('language') or 'n/a'}, license {lic}, "
                 f"last push {str(it.get('pushed_at', ''))[:10]}, {it.get('open_issues_count', 0):,} open issues.")
        out.append(Evidence(f"{it.get('full_name')} (GitHub)", it.get("html_url", ""), truncate(facts, 600),
                            tool="github", query=query, page_text=facts, status="fetched:api",
                            published=str(it.get("pushed_at", ""))[:10]))
    return [e for e in out if e.url]


# --------------------------------------------------------------------------- page reader


def is_public_url(url: str) -> tuple[bool, str]:
    """SSRF guard: only allow http(s) URLs whose host resolves to public IP addresses."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False, "invalid-url"
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, ValueError):
        return False, "dns-failure"
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            return False, "bad-address"
        if not ip.is_global:
            return False, "private-address"
    return True, "ok"


_DATE_META = (
    ("property", "article:published_time"), ("property", "og:published_time"), ("name", "date"),
    ("name", "pubdate"), ("name", "publish-date"), ("itemprop", "datePublished"), ("name", "dc.date"),
    ("property", "article:modified_time"), ("name", "last-modified"),
)
_LD_DATE = re.compile(r'"date(?:Published|Modified)"\s*:\s*"(\d{4}-\d{2}-\d{2})')


def extract_text(html: bytes | str) -> tuple[str, str]:
    """Readable main text and publication date (YYYY-MM-DD or '') from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    published = ""
    for attr, value in _DATE_META:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            published = str(tag["content"])[:10]
            break
    if not published:
        m = _LD_DATE.search(str(soup))
        if m:
            published = m.group(1)
    if not published:
        t = soup.find("time", attrs={"datetime": True})
        if t:
            published = str(t["datetime"])[:10]
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = clean(root.get_text(" ", strip=True))
    if len(text) < 300 and soup.body is not None and root is not soup.body:
        text = clean(soup.body.get_text(" ", strip=True))
    return text, published if re.match(r"\d{4}", published) else ""


def fetch_page(url: str) -> tuple[str, str, str]:
    """Return (text, status, published). Never raises; the status explains failures."""
    current = url
    for _ in range(4):  # follow at most 3 redirects, re-checking every hop
        ok, reason = is_public_url(current)
        if not ok:
            return "", f"blocked:{reason}", ""
        try:
            with requests.get(current, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                              stream=True, allow_redirects=False) as r:
                if r.is_redirect and r.headers.get("location"):
                    current = urljoin(current, r.headers["location"])
                    continue
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").lower()
                if not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                    return "", "skipped-non-text", ""
                raw = b""
                for chunk in r.iter_content(65536):
                    raw += chunk
                    if len(raw) > MAX_PAGE_BYTES:
                        break
                status = f"fetched:{r.status_code}"
                if "text/plain" in ctype:
                    return clean(raw.decode(r.encoding or "utf-8", errors="replace"))[:PAGE_CHARS], status, ""
                text, published = extract_text(raw)
                return text[:PAGE_CHARS], status, published
        except requests.RequestException as ex:
            return "", f"fetch-error:{type(ex).__name__}", ""
    return "", "fetch-error:TooManyRedirects", ""


# --------------------------------------------------------------------------- toolbox

SEARCH_FUNCS = {
    "web_search": web_search,
    "news_search": news_search,
    "wikipedia": wikipedia_search,
    "arxiv": arxiv_search,
    "github": github_search,
}


class ToolBox:
    """Registry the controller calls through; adds caching, timing and offline mode."""

    def __init__(self, enabled: list[str] | tuple[str, ...] | None = None, offline_mode: bool = False,
                 pace: float = 0.0):
        names = list(enabled) if enabled is not None else list(TOOL_SPECS)
        self.enabled = [n for n in TOOL_SPECS if n in names]
        if not any(n in SEARCH_TOOLS for n in self.enabled):
            self.enabled.insert(0, "web_search")
        self.offline = offline_mode
        self.pace = pace  # artificial delay per offline call, so demos are watchable
        self._cache: dict[tuple[str, str], list[Evidence]] = {}

    def available(self, name: str) -> bool:
        return name in self.enabled

    def search(self, tool: str, query: str) -> tuple[list[Evidence], float]:
        if tool not in SEARCH_FUNCS or not self.available(tool):
            raise ToolError(f"unknown-or-disabled-tool:{tool}")
        key = (tool, query.strip().lower())
        start = time.monotonic()
        if key not in self._cache:
            if self.offline:
                time.sleep(self.pace)
            results = offline.search(tool, query) if self.offline else SEARCH_FUNCS[tool](query)
            self._cache[key] = results
        seen, out = set(), []
        for e in self._cache[key]:
            norm = normalize_url(e.url)
            if norm not in seen:
                seen.add(norm)
                out.append(Evidence.from_dict(e.to_dict()))  # copies, so cached results stay pristine
        return out, time.monotonic() - start

    def fetch(self, url: str) -> tuple[str, str, str, float]:
        start = time.monotonic()
        if self.offline:
            time.sleep(self.pace)
        text, status, published = offline.fetch(url) if self.offline else fetch_page(url)
        return text, status, published, time.monotonic() - start

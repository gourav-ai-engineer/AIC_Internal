"""Live tool parsers tested against recorded-shape API responses (HTTP is mocked; no network)."""
import unittest
from unittest.mock import MagicMock, patch

from proofpilot import tools
from proofpilot.tools import ToolError


def response(json_data=None, content=b"", status=200, headers=None):
    r = MagicMock(status_code=status, content=content, headers=headers or {})
    r.json.return_value = json_data
    r.raise_for_status.return_value = None
    return r


ARXIV_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Small Language Models for
      Multilingual Education</title>
    <summary>We study small models for low-resource languages.</summary>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
  </entry>
</feed>"""


class LiveToolParsingTests(unittest.TestCase):
    def test_wikipedia(self):
        data = {"query": {"pages": {
            "2": {"index": 2, "title": "Second", "fullurl": "https://en.wikipedia.org/wiki/Second", "extract": "B."},
            "1": {"index": 1, "title": "First", "fullurl": "https://en.wikipedia.org/wiki/First",
                  "extract": "First is a thing."},
        }}}
        with patch("proofpilot.tools.requests.get", return_value=response(data)):
            out = tools.wikipedia_search("first")
        self.assertEqual([e.title for e in out], ["First — Wikipedia", "Second — Wikipedia"])
        self.assertTrue(out[0].read)
        self.assertEqual(out[0].status, "fetched:api")

    def test_arxiv(self):
        with patch("proofpilot.tools.requests.get", return_value=response(content=ARXIV_XML)):
            out = tools.arxiv_search("small language models education")
        self.assertEqual(len(out), 1)
        e = out[0]
        self.assertEqual(e.title, "Small Language Models for Multilingual Education (arXiv)")
        self.assertTrue(e.url.startswith("https://arxiv.org/abs/"))
        self.assertEqual(e.published, "2024-01-02")
        self.assertIn("A. Author, B. Author", e.page_text)

    def test_arxiv_relaxes_query_when_empty(self):
        empty = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        with patch("proofpilot.tools.requests.get",
                   side_effect=[response(content=empty), response(content=ARXIV_XML)]) as get:
            out = tools.arxiv_search("small language models education")
        self.assertEqual(len(out), 1)
        self.assertIn(" OR ", get.call_args_list[1].kwargs["params"]["search_query"])

    def test_github_falls_back_to_or_query(self):
        item = {"full_name": "org/repo", "html_url": "https://github.com/org/repo", "description": "Agents",
                "stargazers_count": 1234, "forks_count": 5, "language": "Python", "license": {"spdx_id": "MIT"},
                "pushed_at": "2026-09-01T00:00:00Z", "open_issues_count": 3}
        with patch("proofpilot.tools.requests.get",
                   side_effect=[response({"items": []}), response({"items": [item]})]) as get:
            out = tools.github_search("LangGraph vs CrewAI vs AutoGen")
        self.assertEqual(get.call_args_list[1].kwargs["params"]["q"], "langgraph OR crewai OR autogen")
        self.assertIn("1,234 stars", out[0].snippet)
        self.assertEqual(out[0].published, "2026-09-01")

    def test_http_errors_become_tool_errors(self):
        import requests
        with patch("proofpilot.tools.requests.get", side_effect=requests.ConnectionError("down")):
            with self.assertRaises(ToolError):
                tools.wikipedia_search("x")

    def test_web_and_news_search_parse_ddgs(self):
        fake = MagicMock()
        fake.__enter__.return_value = fake
        fake.text.return_value = [{"title": "T", "href": "https://a.org/x", "body": "Body"},
                                  {"title": "Bad", "href": "javascript:alert(1)", "body": ""}]
        fake.news.return_value = [{"title": "N", "url": "https://news.org/n", "body": "B",
                                   "date": "2026-10-01T10:00:00", "source": "News Org"}]
        with patch("ddgs.DDGS", return_value=fake):
            web = tools.web_search("q")
            news = tools.news_search("q")
        self.assertEqual([e.url for e in web], ["https://a.org/x"])
        self.assertEqual(news[0].title, "N (News Org)")
        self.assertEqual(news[0].published, "2026-10-01")

    def test_ddgs_exception_becomes_tool_error(self):
        fake = MagicMock()
        fake.__enter__.return_value = fake
        fake.text.side_effect = RuntimeError("ratelimit")
        with patch("ddgs.DDGS", return_value=fake):
            with self.assertRaises(ToolError):
                tools.web_search("q")

    def test_fetch_page_reads_html(self):
        html = b"<html><body><main><p>" + b"Readable text here. " * 40 + b"</p></main></body></html>"
        r = MagicMock(is_redirect=False, status_code=200, headers={"content-type": "text/html; charset=utf-8"})
        r.__enter__ = lambda s: s
        r.__exit__ = lambda *a: False
        r.iter_content.return_value = [html]
        with patch("proofpilot.tools.is_public_url", return_value=(True, "ok")), \
                patch("proofpilot.tools.requests.get", return_value=r):
            text, status, _ = tools.fetch_page("https://public.org/page")
        self.assertEqual(status, "fetched:200")
        self.assertIn("Readable text here.", text)

    def test_fetch_page_skips_pdf(self):
        r = MagicMock(is_redirect=False, status_code=200, headers={"content-type": "application/pdf"})
        r.__enter__ = lambda s: s
        r.__exit__ = lambda *a: False
        with patch("proofpilot.tools.is_public_url", return_value=(True, "ok")), \
                patch("proofpilot.tools.requests.get", return_value=r):
            self.assertEqual(tools.fetch_page("https://public.org/x.pdf")[1], "skipped-non-text")


if __name__ == "__main__":
    unittest.main()

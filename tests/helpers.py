"""Shared test doubles: a scripted LLM and a scripted toolbox (no network anywhere)."""
from __future__ import annotations

import json
import tempfile

from proofpilot.agent import AgentConfig, ResearchAgent
from proofpilot.llm import LLMError
from proofpilot.memory import Memory
from proofpilot.models import Evidence
from proofpilot.tools import ToolBox, ToolError


class FakeLLM:
    """Answers planning, decision and synthesis prompts with scripted JSON."""

    label = "Fake · test-model"

    def __init__(self, decisions=None, fail=False, plan=None, synthesis=None, critic=None, qa=None):
        self.decisions = list(decisions or [])
        self.critic = critic
        self.qa = qa
        self.fail = fail
        self.plan = plan
        self.synthesis = synthesis
        self.calls = 0
        self.prompts: list[str] = []

    def complete_json(self, prompt: str, **_):
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail:
            raise LLMError("HTTP 429")
        if "planning module" in prompt:
            return self.plan or {"subquestions": [
                {"question": "What are vector databases?", "role": "core", "tools": ["wikipedia"],
                 "queries": ["vector database"]},
                {"question": "Which vector databases are popular?", "role": "core", "tools": ["web_search"],
                 "queries": ["popular vector databases"]},
                {"question": "What are their limitations?", "role": "skeptic", "tools": ["web_search"],
                 "queries": ["vector database limitations"]},
            ]}
        if "controller of ProofPilot" in prompt:
            if self.decisions:
                return self.decisions.pop(0)
            return {"thought": "Enough evidence.", "action": "finish"}
        if "writing module" in prompt:
            return self.synthesis or {
                "executive_answer": "Vector databases store embeddings for similarity search [S1][S99].",
                "subquestion_answers": [{"id": "Q1", "answer": "They index embeddings [S1]."}],
                "findings": [
                    {"claim": "Vector databases store embeddings for similarity search", "sources": ["S1", "S2"]},
                    {"claim": "Quantum teleportation powers every database on Mars", "sources": ["S1"]},
                    {"claim": "Something with a fabricated citation about embeddings", "sources": ["S42"]},
                ],
                "comparison": {"columns": ["Option", "Strength"], "rows": [["A", "fast [S1]"], ["B", "cheap"]]},
                "next_actions": ["Benchmark two options."], "unknowns": ["Pricing."],
                "follow_up_questions": ["Which is cheapest?"],
            }
        if "Critic agent" in prompt:
            return self.critic or {"overall": "One finding needs confirmation.", "objections": [
                {"kind": "single_source", "reason": "C1 needs a second source.", "target": "C1",
                 "query": "vector database embeddings similarity", "tool": "web_search"}]}
        if "contradict each other" in prompt:
            return {"pairs": []}
        if "follow-up question" in prompt:
            return self.qa or {"answerable": True, "answer": "Vector databases store embeddings [S1]."}
        raise AssertionError("unexpected prompt")


SAMPLE = "Vector databases store embeddings for similarity search. They are used in retrieval augmented generation."


def make_results(tool: str, query: str, n: int = 3) -> list[Evidence]:
    return [Evidence(f"{query} result {i}", f"https://site{i}-{tool}.org/{abs(hash(query)) % 997}/{i}",
                     SAMPLE, tool=tool, query=query) for i in range(n)]


class FakeToolBox(ToolBox):
    """Deterministic tools; `failing` tools always raise."""

    def __init__(self, enabled=None, failing=(), empty=(), page=SAMPLE * 3):
        super().__init__(enabled)
        self.failing, self.empty, self.page = set(failing), set(empty), page
        self.search_calls: list[tuple[str, str]] = []
        self.fetch_calls: list[str] = []

    def search(self, tool, query):
        self.search_calls.append((tool, query))
        if tool in self.failing:
            raise ToolError("ConnectionError")
        return ([] if tool in self.empty else make_results(tool, query)), 0.01

    def fetch(self, url):
        self.fetch_calls.append(url)
        return self.page, "fetched:200", "2026-01-10", 0.01


def make_agent(llm=None, toolbox=None, memory_dir=None, **config) -> ResearchAgent:
    config.setdefault("use_memory", memory_dir is not None)
    memory = Memory(memory_dir) if memory_dir else None
    return ResearchAgent(AgentConfig(**config), llm=llm, toolbox=toolbox or FakeToolBox(), memory=memory)


def tempdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory()


def dumps(obj) -> str:
    return json.dumps(obj)

"""Minimal, provider-agnostic LLM client (plain HTTP, no SDK dependencies).

Supported: Google Gemini (free tier) and any OpenAI-compatible endpoint
(OpenAI, Groq, OpenRouter, a local Ollama server). The agent never requires an LLM;
it falls back to deterministic heuristics whenever this client is absent or failing.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests


class LLMError(Exception):
    pass


PROVIDERS = {
    "gemini": {"model": "gemini-2.5-flash"},
    "openai": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "groq": {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "ollama": {"model": "llama3.1", "base_url": "http://localhost:11434/v1"},
}


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str | None = None, base_url: str | None = None,
                 timeout: float = 40):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider = provider
        self.api_key = api_key
        self.model = model or PROVIDERS[provider]["model"]
        self.base_url = (base_url or PROVIDERS[provider].get("base_url", "")).rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.failures = 0

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        env = os.environ
        if env.get("PROOFPILOT_LLM", "").lower() in ("off", "none", "0", "false"):
            return None
        if key := (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip():
            return cls("gemini", key, env.get("GEMINI_MODEL"))
        if key := env.get("GROQ_API_KEY", "").strip():
            return cls("groq", key, env.get("GROQ_MODEL"))
        if key := env.get("OPENAI_API_KEY", "").strip():
            return cls("openai", key, env.get("OPENAI_MODEL"), env.get("OPENAI_BASE_URL"))
        if model := env.get("OLLAMA_MODEL", "").strip():
            return cls("ollama", "ollama", model, env.get("OLLAMA_BASE_URL"))
        return None

    @property
    def label(self) -> str:
        return f"{self.provider.title()} · {self.model}"

    # ------------------------------------------------------------------ requests

    def _gemini(self, prompt: str, json_mode: bool, temperature: float, max_tokens: int) -> str:
        config: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            config["responseMimeType"] = "application/json"
        if self.model.startswith("gemini-2.5-flash"):
            config["thinkingConfig"] = {"thinkingBudget": 0}  # keep latency and quota low
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},  # header, so the key never appears in URLs/logs
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config},
            timeout=self.timeout,
        )
        self._raise_for_status(r)
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError, ValueError) as ex:
            raise LLMError("unexpected Gemini response") from ex

    def _openai(self, prompt: str, json_mode: bool, temperature: float, max_tokens: int) -> str:
        body: dict = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        self._raise_for_status(r)
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError) as ex:
            raise LLMError("unexpected chat-completions response") from ex

    @staticmethod
    def _raise_for_status(r: requests.Response) -> None:
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}", r.status_code)

    def complete(self, prompt: str, *, json_mode: bool = False, temperature: float = 0.2,
                 max_tokens: int = 1500, retries: int = 1) -> str:
        """Send one prompt; retry once on rate limits / server errors / network failures."""
        send = self._gemini if self.provider == "gemini" else self._openai
        for attempt in range(retries + 1):
            self.calls += 1
            try:
                text = send(prompt, json_mode, temperature, max_tokens)
                if not text.strip():
                    raise LLMError("empty response")
                return text
            except (requests.RequestException, LLMError) as ex:
                code = ex.args[1] if isinstance(ex, LLMError) and len(ex.args) > 1 else None
                retryable = code is None or code == 429 or code >= 500
                if attempt < retries and retryable:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                self.failures += 1
                reason = str(ex.args[0]) if isinstance(ex, LLMError) and ex.args else type(ex).__name__
                raise LLMError(reason) from ex
        raise LLMError("unreachable")

    def complete_json(self, prompt: str, **kwargs) -> dict:
        return parse_json(self.complete(prompt, json_mode=True, **kwargs))


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response (tolerates code fences and chatter)."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("no JSON object in response")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as ex:
        raise LLMError("invalid JSON in response") from ex
    if not isinstance(data, dict):
        raise LLMError("JSON response is not an object")
    return data

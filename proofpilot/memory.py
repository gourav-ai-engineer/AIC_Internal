"""Long-term memory: every run is saved, and related past runs are recalled.

When a new goal resembles an earlier one, the agent starts from the trustworthy
sources it already found instead of from zero.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .textutil import content_terms, jaccard


def default_home() -> Path:
    return Path(os.getenv("PROOFPILOT_HOME") or Path.home() / ".proofpilot")


class Memory:
    def __init__(self, root: Path | str | None = None):
        self.dir = Path(root or default_home()) / "runs"

    def save(self, result: dict) -> str | None:
        """Persist a run; returns its file id, or None if the disk is not writable."""
        slug = re.sub(r"[^a-z0-9]+", "-", result.get("goal", "run").lower()).strip("-")[:40] or "run"
        run_id = f"{result.get('run_id', 'run')}-{slug}"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{run_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
            return run_id
        except OSError:
            return None

    def _files(self) -> list[Path]:
        try:
            return sorted(self.dir.glob("*.json"), reverse=True)
        except OSError:
            return []

    def load(self, run_id: str) -> dict | None:
        path = self.dir / f"{Path(run_id).name}.json"
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def history(self, limit: int = 20) -> list[dict]:
        out = []
        for path in self._files()[:limit]:
            try:
                data = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": path.stem, "goal": data.get("goal", ""), "created": data.get("created", ""),
                        "evidence_strength": data.get("metrics", {}).get("evidence_strength")})
        return out

    def recall(self, goal: str, threshold: float = 0.3, limit: int = 2) -> list[dict]:
        """Past runs whose goals overlap with this one, most similar first."""
        terms = content_terms(goal)
        matches = []
        for path in self._files()[:200]:
            try:
                data = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            sim = jaccard(terms, content_terms(data.get("goal", "")))
            if sim >= threshold:
                matches.append((sim, path.stem, data))
        matches.sort(key=lambda m: m[0], reverse=True)
        return [{"id": rid, "similarity": round(sim, 2), "goal": d.get("goal", ""), "created": d.get("created", ""),
                 "mode": d.get("mode", ""), "evidence": d.get("evidence", [])} for sim, rid, d in matches[:limit]]

    def clear(self) -> int:
        removed = 0
        for path in self._files():
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed

"""Command-line interface: `python -m proofpilot "your goal" [--offline]`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import AgentConfig, ResearchAgent
from .tools import TOOL_SPECS

ICONS = {"goal": "🎯", "memory": "🧠", "plan": "🗺️", "decide": "🤔", "act": "🔧", "observe": "👁️", "adapt": "🔁",
         "guardrail": "🛡️", "reflect": "🪞", "synthesize": "✍️", "verify": "✅", "report": "📄"}


def _print_event(e: dict) -> None:
    print(f"[{e['t']:6.1f}s] {ICONS.get(e['phase'], '•')} {e['phase'].upper():<10} {e['message']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not support emoji
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="proofpilot", description="Evidence-first agentic research from the terminal.")
    ap.add_argument("goal", help="what you want researched")
    ap.add_argument("--budget", type=int, default=14, help="maximum tool calls (default 14)")
    ap.add_argument("--time-limit", type=float, default=150, help="seconds before the loop stops (default 150)")
    ap.add_argument("--policy", choices=["auto", "llm", "heuristic"], default="auto",
                    help="who chooses the next action (default: LLM if a key is set, else heuristic)")
    ap.add_argument("--tools", default=",".join(TOOL_SPECS), help="comma-separated tools to enable")
    ap.add_argument("--offline", action="store_true", help="use synthetic demo sources (no network)")
    ap.add_argument("--no-memory", action="store_true", help="do not recall or save past runs")
    ap.add_argument("--out", type=Path, help="write the Markdown brief to this file")
    ap.add_argument("--json", type=Path, help="write the full run (trace, evidence, claims) as JSON")
    ap.add_argument("--quiet", action="store_true", help="do not stream the agent trace")
    args = ap.parse_args(argv)

    tools = tuple(t.strip() for t in args.tools.split(",") if t.strip() in TOOL_SPECS)
    config = AgentConfig(budget=args.budget, time_limit=args.time_limit, policy=args.policy, tools=tools,
                         offline=args.offline, use_memory=not args.no_memory)
    agent = ResearchAgent(config, progress=None if args.quiet else _print_event)
    try:
        result = agent.run(args.goal)
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2
    if args.out:
        args.out.write_text(result["markdown"], "utf-8")
        print(f"\nBrief written to {args.out}")
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
        print(f"Run data written to {args.json}")
    if not args.out:
        print("\n" + result["markdown"])
    return 0

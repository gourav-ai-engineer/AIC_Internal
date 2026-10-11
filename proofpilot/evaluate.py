"""Measure the verifier on hand-labelled cases: `python -m proofpilot.evaluate`.

Reports precision / recall / accuracy for (1) claim grounding — "does this source support
this claim?" — and (2) contradiction detection — "do these two statements conflict?".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conflicts import compare
from .verify import GROUNDED, support_score

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.json"


def _scores(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(p and t for p, t in pairs)
    fp = sum(p and not t for p, t in pairs)
    fn = sum(not p and t for p, t in pairs)
    tn = sum(not p and not t for p, t in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"cases": len(pairs), "accuracy": round((tp + tn) / len(pairs), 3) if pairs else 0.0,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run(path: Path = DEFAULT_CASES) -> dict:
    data = json.loads(Path(path).read_text("utf-8"))
    grounding, g_errors = [], []
    for c in data["grounding"]:
        score = support_score(c["claim"], c["source"])
        predicted = score >= GROUNDED
        grounding.append((predicted, c["supported"]))
        if predicted != c["supported"]:
            g_errors.append({**c, "score": score})
    conflict, c_errors = [], []
    for c in data["conflict"]:
        predicted = compare(c["a"], c["b"]) is not None
        conflict.append((predicted, c["contradict"]))
        if predicted != c["contradict"]:
            c_errors.append(c)
    return {"grounding": _scores(grounding), "conflict": _scores(conflict),
            "grounding_errors": g_errors, "conflict_errors": c_errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate ProofPilot's verifier on labelled cases.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args(argv)
    r = run(args.cases)
    for name in ("grounding", "conflict"):
        s = r[name]
        print(f"{name:<10} cases={s['cases']:<3} accuracy={s['accuracy']:.0%} precision={s['precision']:.0%} "
              f"recall={s['recall']:.0%} f1={s['f1']:.2f}  (tp={s['tp']} fp={s['fp']} fn={s['fn']} tn={s['tn']})")
    if args.show_errors:
        for e in r["grounding_errors"]:
            print(f"  grounding miss (score {e['score']}): {e['claim']!r} vs {e['source'][:70]!r} "
                  f"-> expected {e['supported']}")
        for e in r["conflict_errors"]:
            print(f"  conflict miss: {e['a']!r} vs {e['b']!r} -> expected {e['contradict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

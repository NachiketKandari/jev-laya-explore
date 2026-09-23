"""Classify Banking77 demo queries with Laya and save results for the web app.

Reads data/banking77_test.csv + data/coarse_labels.json, classifies the first
N rows with LayaClassifier, writes web/public/results.json (also read by the
notebook's last cell).

    uv run scripts/classify_demo_data.py [--n 300] [--runtime auto] [--out web/public/results.json]

~40 ms/doc on MLX, so 300 rows takes ~15 s plus model load.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("USE_TF", "0")

from laya_classify import LayaClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--runtime", default="auto")
    ap.add_argument("--out", default="web/public/results.json")
    args = ap.parse_args()

    labels: dict = json.loads((ROOT / "data/coarse_labels.json").read_text())
    rows = list(csv.DictReader(open(ROOT / "data/banking77_test.csv", encoding="utf-8")))
    # test.csv is grouped by intent, so first-N would cover 2-3 groups only.
    # Seeded shuffle -> deterministic stratified-ish sample, same rows every run.
    random.Random(7).shuffle(rows)
    rows = rows[: args.n]

    clf = LayaClassifier(labels, runtime=args.runtime)
    clf.warm()
    out_rows, lat, correct = [], [], 0
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        pred = clf.predict(r["text"])
        ok = pred.label == r["coarse_label"]
        correct += ok
        lat.append(pred.latency_ms)
        out_rows.append({
            "text": r["text"], "fine": r["fine_label"], "gold": r["coarse_label"],
            "pred": pred.label, "confidence": round(pred.confidence, 4),
            "correct": ok, "probabilities": {k: round(v, 4) for k, v in pred.probabilities.items()},
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} ...", flush=True)

    per_class = {}
    for label in labels:
        gold_idx = [x for x in out_rows if x["gold"] == label]
        per_class[label] = {
            "n": len(gold_idx),
            "correct": sum(x["correct"] for x in gold_idx),
            "accuracy": round(sum(x["correct"] for x in gold_idx) / max(1, len(gold_idx)), 4),
        }
    confusions = [[g, p, n] for (g, p), n in
                  Counter((x["gold"], x["pred"]) for x in out_rows if not x["correct"]).most_common(10)]
    right = [x["confidence"] for x in out_rows if x["correct"]]
    wrong = [x["confidence"] for x in out_rows if not x["correct"]]

    result = {
        "meta": {
            "dataset": "PolyAI Banking77 test, mapped to 7 coarse groups (CC-BY-4.0)",
            "n": len(out_rows), "runtime": clf.runtime,
            "accuracy": round(correct / len(out_rows), 4),
            "mean_latency_ms": round(sum(lat) / len(lat), 1),
            "total_s": round(time.perf_counter() - t0, 1),
            "labels": labels,
        },
        "per_class": per_class,
        "confusions": confusions,
        "confidence": {
            "mean_when_right": round(sum(right) / max(1, len(right)), 4),
            "mean_when_wrong": round(sum(wrong) / max(1, len(wrong)), 4),
        },
        "rows": out_rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"accuracy {correct}/{len(out_rows)} = {correct / len(out_rows):.3f} -> {out}")


if __name__ == "__main__":
    main()

"""Measure real Jev (typesafe/jev-1.13 via OpenRouter decisions endpoint).

Same 300-row seed-7 sample and same 7 criteria Laya gets. Key is read at
runtime from $OPENROUTER_API_KEY or ../../idirect-playwright/.env and is
NEVER logged or written anywhere.

    uv run scripts/bench_jev.py [--n 300] [--seed 7]

~250 ms/call, ~$0.00002/call -> 300 rows costs about half a cent.
Writes benchmarks/jev_banking77_coarse7.json (predictions only, no key).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_file = ROOT.parent / "idirect-playwright" / ".env"
    for line in env_file.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no OpenRouter key: set $OPENROUTER_API_KEY")


def call_jev(key: str, text: str, labels: dict, instructions: str) -> tuple[dict, float]:
    body = json.dumps({
        "model": MODEL,
        "state": {"message": text},
        "questions": {"label": {"type": "choice", "instructions": instructions,
                                "criteria": labels}},
    }).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            return resp, (time.perf_counter() - t0) * 1000
        except Exception as e:  # transient (429/5xx/timeout) -> back off, retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"jev failed 3x: {type(last).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="benchmarks/jev_banking77_coarse7.json")
    args = ap.parse_args()

    key = load_key()
    labels: dict = json.loads((ROOT / "data/coarse_labels.json").read_text())
    instructions = "Which team should handle the request in `message`?"
    rows = list(csv.DictReader(open(ROOT / "data/banking77_test.csv", encoding="utf-8")))
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]

    out_rows, lat, correct, cost = [], [], 0, 0.0
    model_seen = None
    for i, r in enumerate(sample):
        resp, ms = call_jev(key, r["text"], labels, instructions)
        ans = resp["answers"]["label"]
        ok = ans["choice"] == r["coarse_label"]
        correct += ok
        lat.append(ms)
        cost += float(resp.get("usage", {}).get("cost", 0.0))
        model_seen = resp.get("model")
        out_rows.append({"text": r["text"], "fine": r["fine_label"],
                         "gold": r["coarse_label"], "pred": ans["choice"],
                         "confidence": ans["confidence"], "correct": ok,
                         "probabilities": ans.get("probabilities", {})})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)} acc={correct / (i + 1):.3f} "
                  f"cost=${cost:.4f}", flush=True)
        time.sleep(0.1)

    lat_sorted = sorted(lat)
    result = {
        "meta": {"model": model_seen, "endpoint": "openrouter /api/alpha/decisions",
                 "n": len(sample), "seed": args.seed,
                 "accuracy": round(correct / len(sample), 4),
                 "p50_ms": round(lat_sorted[len(lat_sorted) // 2], 1),
                 "mean_ms": round(sum(lat) / len(lat), 1),
                 "total_cost_usd": round(cost, 5),
                 "labels": "same 7 coarse groups + full descriptions as Laya"},
        "rows": out_rows,
    }
    out = ROOT / args.out
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(f"jev acc={correct}/{len(sample)}={correct / len(sample):.3f} "
          f"p50={result['meta']['p50_ms']:.0f}ms cost=${cost:.4f} -> {out}")


if __name__ == "__main__":
    main()

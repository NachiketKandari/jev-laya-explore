"""Head-to-head on the same Banking77 coarse-7 sample: Laya vs TF-IDF+LogReg.

Jev is included as *cited* numbers only (hosted API, needs a key; see COMPARISON.md
for the recipe to measure it yourself).

    uv run --with scikit-learn scripts/bench_comparison.py [--n 300] [--seed 7]

Writes benchmarks/banking77_coarse7.json. Laya ~25 s for 300 docs on MLX;
sklearn trains on 10k rows in seconds on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("USE_TF", "0")

from fetch_demo_data import FINE_TO_COARSE  # noqa: E402
from laya_classify import LayaClassifier  # noqa: E402


def pct(values, p):
    s = sorted(values)
    return s[min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)]


def load_rows(path: str) -> list:
    return list(csv.DictReader(open(ROOT / path, encoding="utf-8")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runtime", default="auto")
    ap.add_argument("--out", default="benchmarks/banking77_coarse7.json")
    args = ap.parse_args()

    labels: dict = json.loads((ROOT / "data/coarse_labels.json").read_text())
    test = load_rows("data/banking77_test.csv")
    random.Random(args.seed).shuffle(test)
    sample = test[: args.n]
    texts = [r["text"] for r in sample]
    gold = [r["coarse_label"] for r in sample]

    # --- Laya: zero-shot, descriptions are the whole model -------------------
    clf = LayaClassifier(labels, runtime=args.runtime)
    clf.warm()
    laya_preds, laya_lat = [], []
    for t in texts:
        p = clf.predict(t)
        laya_preds.append(p.label)
        laya_lat.append(p.latency_ms)
    laya_acc = sum(a == b for a, b in zip(laya_preds, gold)) / len(gold)

    # --- sklearn: TF-IDF + logistic regression, trained on 10k rows ---------
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError:
        sys.exit("needs scikit-learn: uv run --with scikit-learn scripts/bench_comparison.py")
    train = load_rows("data/banking77_train.csv")
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50_000),
        LogisticRegression(max_iter=1000),
    )
    t0 = time.perf_counter()
    model.fit([r["text"] for r in train], [r["coarse_label"] for r in train])
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_preds = list(model.predict(texts))
    batch_ms = (time.perf_counter() - t0) * 1000 / len(texts)
    single_lat = []
    for t in texts:
        t1 = time.perf_counter()
        model.predict([t])
        single_lat.append((time.perf_counter() - t1) * 1000)
    sk_acc = sum(a == b for a, b in zip(batch_preds, gold)) / len(gold)

    result = {
        "meta": {
            "dataset": "PolyAI Banking77 test->7 coarse groups; sklearn trained on train->7",
            "n": len(sample), "seed": args.seed, "machine": "M1 Pro 16GB",
            "note": "jev figures are published by others, NOT measured here",
        },
        "laya": {
            "variant": f"english checkpoint, {clf.runtime} runtime, zero-shot (no training)",
            "accuracy": round(laya_acc, 4),
            "p50_ms": round(pct(laya_lat, 50), 1), "p95_ms": round(pct(laya_lat, 95), 1),
            "mean_ms": round(statistics.mean(laya_lat), 1),
            "train_rows": 0, "weights": "421M vendored, Apache-2.0",
        },
        "sklearn_tfidf_logreg": {
            "variant": "TfidfVectorizer(1-2gram, 50k) + LogisticRegression, trained",
            "accuracy": round(sk_acc, 4),
            "train_s": round(train_s, 1), "train_rows": len(train),
            "predict_single_p50_ms": round(pct(single_lat, 50), 3),
            "predict_batch_ms_per_doc": round(batch_ms, 4),
        },
        "jev_cited": {
            "measured_here": False,
            "accuracy_banking77_full_77labels": 0.870,
            "accuracy_ag_news_4labels": 0.910,
            "accuracy_dair_emotion_6labels": 0.480,
            "p50_ms": "236-276",
            "cost": "$0.042 / 1M tokens, closed API",
            "sources": [
                "https://github.com/NandhaKishorM/laya (README + BENCHMARKS.md table)",
                "https://github.com/AbdelStark/jev-benchmarks (latency)",
                "https://github.com/nibzard/decision-model-benchmark (latency)",
            ],
        },
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"n={len(sample)} seed={args.seed}")
    print(f"  laya    acc={laya_acc:.3f}  p50={pct(laya_lat, 50):.1f}ms p95={pct(laya_lat, 95):.1f}ms  train=none")
    print(f"  sklearn acc={sk_acc:.3f}  train={train_s:.1f}s/{len(train)}rows  "
          f"single={pct(single_lat, 50):.3f}ms/doc batch={batch_ms:.4f}ms/doc")
    print(f"  jev     acc=0.870 (CITED, full 77-label Banking77)  p50=236-276ms (CITED)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

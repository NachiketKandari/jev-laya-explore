"""Head-to-head: the two Laya runtimes that work on this Mac.

  * `laya`     - PyTorch, runs on the Apple GPU via MPS
  * `laya-mlx` - native MLX port, FP16, no PyTorch in the loop

Same checkpoint family, same questions, same documents. Reports load time,
latency distribution, accuracy, and whether the two runtimes even agree.

    python compare_runtimes.py
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import statistics
import time

from bench_classify import GOLD, pct
from laya_classify import LABELS, INSTRUCTIONS
from laya_paths import describe, model_for

QUESTIONS = {"label": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": LABELS}}


def questions_for(k: int):
    return {
        f"q{i}": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": LABELS}
        for i in range(k)
    }


def run(loader, label: str, n_latency: int = 25) -> dict:
    print(f"\n--- {label} ---")
    t0 = time.perf_counter()
    agent = loader()
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:5.1f} s   dtype={getattr(agent, 'dtype', 'n/a')}  "
          f"device={getattr(agent, 'device', 'n/a')}")

    # Warm up: first call includes graph build / kernel compilation.
    for i in range(3):
        agent.predict({"message": GOLD[i][0]}, QUESTIONS)

    latencies, correct, labels = [], 0, []
    for i in range(n_latency):
        text, gold = GOLD[i % len(GOLD)]
        t = time.perf_counter()
        res = agent.predict({"message": text}, QUESTIONS)
        latencies.append((time.perf_counter() - t) * 1000)
        if i < len(GOLD):
            choice = res["answers"]["label"]["choice"]
            labels.append(choice)
            correct += choice == gold

    batches = {}
    for k in (1, 5):
        ts = []
        for _ in range(6):
            t = time.perf_counter()
            agent.predict({"message": GOLD[0][0]}, questions_for(k))
            ts.append((time.perf_counter() - t) * 1000)
        batches[k] = statistics.mean(ts)

    print(f"  p50 {pct(latencies, 50):.1f} ms | p95 {pct(latencies, 95):.1f} ms | "
          f"mean {statistics.mean(latencies):.1f} ms")
    print(f"  throughput ~{1000 / statistics.mean(latencies):.1f} docs/s single process")
    print(f"  accuracy {correct}/{len(GOLD)} = {correct / len(GOLD):.3f}")
    print(f"  1 question {batches[1]:.1f} ms | 5 questions {batches[5]:.1f} ms")
    return {"label": label, "load_s": load_s, "p50": pct(latencies, 50),
            "p95": pct(latencies, 95), "mean": statistics.mean(latencies),
            "accuracy": correct / len(GOLD), "labels": labels}


def main() -> None:
    import laya
    import laya_mlx

    print(f"  {describe()}")
    results = [
        run(lambda: laya.load(*model_for()), "PyTorch (MPS)"),
        run(lambda: laya_mlx.load(*model_for(mlx=True)), "MLX (FP16)"),
    ]

    print(f"\n{'=' * 66}\nSummary (Apple M1 Pro, 16 GB, one doc per call)\n{'=' * 66}")
    print(f"  {'runtime':<16}{'load':>8}{'p50':>9}{'p95':>9}{'docs/s':>9}{'acc':>7}")
    for r in results:
        print(f"  {r['label']:<16}{r['load_s']:>7.1f}s{r['p50']:>8.1f}m{r['p95']:>8.1f}m"
              f"{1000 / r['mean']:>9.1f}{r['accuracy']:>7.3f}")

    a, b = results
    if a["labels"] and b["labels"]:
        agree = sum(x == y for x, y in zip(a["labels"], b["labels"]))
        print(f"\n  runtime agreement: {agree}/{len(a['labels'])} identical labels")
    speedup = a["p50"] / b["p50"]
    print(f"  MLX is {speedup:.2f}x faster at p50 ({a['p50']:.1f} ms -> {b['p50']:.1f} ms)")


if __name__ == "__main__":
    main()

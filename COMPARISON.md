# Laya vs Jev vs TF-IDF+LogReg on Banking77 coarse-7

Same 300 queries (seed 7), same 7 labels, M1 Pro 16 GB. Raw numbers live in
`benchmarks/banking77_coarse7.json` — regenerate with:

```bash
uv run --with scikit-learn scripts/bench_comparison.py
```

(`--with` keeps scikit-learn out of `pyproject.toml`; it is a benchmark-only dep.)

## Result

| | Laya (measured) | TF-IDF + LogReg (measured) | Jev (cited, not measured) |
|---|---|---|---|
| accuracy | 0.693 | **0.927** | 0.870* |
| label space | 7 coarse, zero-shot | 7 coarse, trained | *77 full intents* |
| training data | none (7 descriptions) | 10,003 labelled rows, 0.8 s CPU | none (API) |
| latency | p50 51 ms / p95 62 ms | **0.23 ms single, 0.01 ms batch** | 236–276 ms p50 |
| cost | $0 self-hosted, Apache-2.0 | $0, sklearn BSD | $0.042 / 1M tokens, closed |
| calibration | temperatures to fit | `predict_proba` (uncalibrated) | published ECE 0.246 raw |

\* Jev's 0.870 is on the **full 77-label** task (harder per-guess than 7-way, so not
directly comparable) and is **published by others, never run here** — no TypeSafe
key, and it is a paid hosted API. Sources: `NandhaKishorM/laya` README/BENCHMARKS.md
table, latency via `AbdelStark/jev-benchmarks` and `nibzard/decision-model-benchmark`.

## Reading it honestly

- **sklearn wins this task outright** — 0.927 at microseconds per doc. Single
  English domain + 10k labels is exactly what linear models are for. If your
  labels are fixed and you have labels, start here.
- **Laya's 0.693 cost zero labels.** Its edge is elsewhere: new labels without
  retraining, 5 mixed questions (choice+score+noul) in one ~150 ms pass,
  calibrated probabilities, multilingual routing. Fine-tuned it beats Jev on
  typed-decisions (0.766 vs 0.727, upstream, same caveat as above).
- **Jev's edge is >20 options out of the box** (0.870 on 77 labels vs Laya's
  0.425 at default budgets) and zero ops. You pay per token and ship data off-box.

## How to test each (reproduce or extend)

**Laya** — `scripts/bench_comparison.py` (MLX auto, falls back to torch) or the
notebook. Vary `--n`, `--seed`, edit descriptions in `data/coarse_labels.json`.

**sklearn baseline** — same script. Swap the pipeline to try
`LinearSVC`, `ComplementNB`, or char n-grams; add a 77-label run to feel why
Laya's shortlist section exists.

**Jev** — to measure it yourself on this sample: get a TypeSafe key, POST each
`{state: {message}, questions: {label: {type: choice, ...same 7 criteria}}}` to
`/v1/systemone`, and score `answers.label.choice` against `gold`. Keep prompts
byte-identical and report seed, n, and label count — the upstream table mixes
72 vs 77 labels and different samples, which is why this doc refuses to rank Jev
against the two measured columns.

## Tradeoffs

- Fixed labels + labelled data → sklearn (accuracy, latency, simplicity).
- Changing labels, no labels, multilingual, or multi-signal per doc → Laya.
- No infra / 50+ options in one call / someone else's uptime → Jev.
- Need Laya-grade accuracy on *your* schema → fine-tune it (`TRAINING.md`);
  upstream went 0.36 → 0.766 that way.

## Limits of this benchmark

300 rows (10% of test), coarse-7 not the standard 77-label task, Laya zero-shot
only, sklearn English-only, Jev cited. It answers "what should I reach for?"
not "what is SOTA?".

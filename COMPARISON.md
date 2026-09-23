# Laya vs Jev vs TF-IDF+LogReg on Banking77 coarse-7

Same 300 queries (seed 7), same 7 labels, M1 Pro 16 GB. Laya/sklearn/NLI ran
here; **Jev ran for real too** (`typesafe/jev-1.13-20260917` via OpenRouter's
`/api/alpha/decisions`, key from `../idirect-playwright/.env`, ~$0.006 total).
Raw numbers in `benchmarks/banking77_coarse7.json` (+ per-row
`benchmarks/jev_banking77_coarse7.json`) — regenerate with:

```bash
uv run --with scikit-learn scripts/bench_comparison.py
uv run --with scikit-learn scripts/bench_comparison.py --nli   # + bart-large-mnli zero-shot (~1.6GB dl, minutes)
uv run scripts/bench_jev.py                                    # needs $OPENROUTER_API_KEY (never logged)
```

(`--with` keeps scikit-learn out of `pyproject.toml`; it is a benchmark-only dep.)

## Result

| | Laya (measured) | Jev (measured) | TF-IDF + LogReg (measured) | NLI zero-shot (measured) |
|---|---|---|---|---|
| accuracy | 0.693 | 0.793 | **0.927** | 0.533 |
| label space | 7 coarse, zero-shot | 7 coarse, zero-shot | 7 coarse, trained | 7 coarse, zero-shot |
| training data | none (7 descriptions) | none (API) | 10,003 labelled rows, 0.8 s CPU | none |
| latency | p50 51 ms / p95 62 ms | p50 463 ms | **0.23 ms single, 0.01 ms batch** | p50 122 ms (7 passes/doc) |
| cost | $0 self-hosted, Apache-2.0 | $0.021 / 1k calls, closed | $0, sklearn BSD | $0, BART MIT |
| calibration | temperatures to fit | conf 0.94 right / 0.74 wrong; 16 rows zero-prob on gold | `predict_proba` (uncalibrated) | entailment scores (uncalibrated) |

Jev's published 0.870 is on the **full 77-label** task (harder per-guess than
7-way) — on our shared 7-label sample it scores **0.793**, +0.100 over Laya,
at 9× the latency and ~2¢/1k calls. Its per-class best: fees_charges 0.931,
account_identity 0.900; worst: transfers 0.675. The 16 zero-probability-on-gold
rows echo the upstream DAIR-Emotion finding, at 5% here instead of 16%.

## Reading it honestly

- **Jev is the best zero-shot option measured** — 0.793, +0.100 over Laya on
  identical inputs. You pay 463 ms and data-leaves-the-building per call.
- **sklearn wins the task outright** — 0.927 at microseconds per doc. Single
  English domain + 10k labels is exactly what linear models are for. If your
  labels are fixed and you have labels, start here.
- **NLI zero-shot is not "better Laya" here** — bart-large-mnli (407M, same size
  class) scored 0.533 with the same descriptions as hypotheses, at 122 ms/doc
  (one forward pass *per label* vs Laya's one pass for all 7). Laya's
  mask-per-option scoring beats premise-hypothesis entailment on this task.
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

**Jev** — `scripts/bench_jev.py` (same sample, same criteria; key from
`$OPENROUTER_API_KEY` or `../idirect-playwright/.env`, never logged). To extend:
vary `--n`, add `score`/`noul` questions, or run the full 77-label task to check
the published 0.870 yourself.

## Tradeoffs

- Fixed labels + labelled data → sklearn (accuracy, latency, simplicity).
- Changing labels, no labels, multilingual, or multi-signal per doc → Laya.
- No infra / 50+ options in one call / someone else's uptime → Jev.
- Need Laya-grade accuracy on *your* schema → fine-tune it (`TRAINING.md`);
  upstream went 0.36 → 0.766 that way.

## Limits of this benchmark

300 rows (10% of test), coarse-7 not the standard 77-label task, Laya zero-shot
only, sklearn English-only. Jev measured over the network from India (p50
463 ms vs 236–276 ms in US/EU third-party runs — geography + prompt size).
It answers "what should I reach for?" not "what is SOTA?".

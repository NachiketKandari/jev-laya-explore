# Journey: everything tried here, in order (2026-09-23, M1 Pro 16 GB)

One-day lab log: Laya and Jev brought into this workspace, measured against
each other and classic ML, taught via tutorial + notebook + web demo, then
trained locally. Start here; each section links to the doc with the detail.

## Timeline

1. **Setup + first measurements** (`FINDINGS.md`). Installed `laya 0.3.5` and
   `laya-mlx 0.2.0`, found MLX 1.4× faster (p50 42.7 vs 59.9 ms) with 18/18
   identical labels, vendored 3.2 GB of weights + 221 MB wheelhouse, verified
   fully offline. Key early finding: raw confidence does not separate
   right from wrong (0.551 vs 0.543).
2. **Tutorial** (`TUTORIAL.md`, `tutorial_laya.py`). The three primitives —
   `choice`, `noul`, `score` — with live outputs, presets, Router, shortlist.
3. **Real data + notebook + web** (`data/`, `notebooks/`, `web/`). Banking77
   test (3,080 queries) mapped 77 intents → 7 groups; notebook classifies 200
   (0.700, all cells executed); Next.js explorer renders `results.json`.
4. **Three-way benchmark** (`COMPARISON.md`, `benchmarks/`). sklearn TF-IDF +
   LogReg trained on 10k rows: **0.927** at 0.2 ms. bart-large-mnli zero-shot:
   0.533 at 122 ms. Laya zero-shot: 0.693 at 51 ms.
5. **Real Jev** (`scripts/bench_jev.py`). Found the OpenRouter key + verified
   contract in the neighbouring `idirect-playwright` project;
   `typesafe/jev-1.13` scored **0.793** on the identical sample ($0.0064).
6. **Local training** (`TRAINING.md`, `training/`). Measured MPS backward peak
   3.38 GB → training fits 16 GB. Calibration (T=1.2) + head fine-tune
   (8.6 min): NLL/ECE improved, accuracy flat at 0.693.
7. **Second dataset** (`data/finphrasebank_*`). Financial PhraseBank sentiment:
   Laya zero-shot **0.914** (note: CC-BY-NC-SA license).
8. **Product framing** (`PRODUCTS.md`, `demo_llm_couple.py`). BFSI
   data-residency use cases, 3 Laya+LLM patterns, live gate+router demo.

## Experiments and results

| # | experiment | setup | result |
|---|---|---|---|
| 1 | runtime head-to-head | 18 tickets, torch MPS vs MLX | MLX 1.4× faster, 18/18 agree |
| 2 | confidence gating | 18 tickets, thresholds 0.5–0.95 | every gate hurts; ≥0.95 → 0.500 acc |
| 3 | question scaling | 1/3/5 questions per call | batching saves ~25%, questions not free |
| 4 | Banking77 coarse-7, Laya | 300 rows, seed 7 | 0.693 @ p50 51 ms |
| 5 | same, sklearn (trained) | TF-IDF 1-2gram + LogReg, 10k rows | **0.927** @ 0.2 ms, 0.8 s train |
| 6 | same, bart-mnli zero-shot | same descriptions as hypotheses | 0.533 @ 122 ms (7 passes) |
| 7 | same, Jev via OpenRouter | `typesafe/jev-1.13`, identical inputs | **0.793** @ 463 ms, $0.0064 |
| 8 | temperature fit | T on 150, held-out 150 | T=1.2, NLL 1.069→1.041 |
| 9 | head fine-tune | frozen encoder, 3000×2 epochs, MPS | NLL 0.995→0.927, ECE →0.080, acc flat |
| 10 | PhraseBank sentiment | 453 test, zero-shot | **0.914** |
| 11 | LLM coupling demo | guard preset + router, stub LLM | jailbreak 1.0 blocked pre-LLM |

## Discoveries (the durable ones)

1. MLX ≈ 1.4× PyTorch with identical labels — use it on Apple Silicon.
2. Do not gate on raw confidence until temperatures are fitted on your data.
3. Descriptions are the training set; label count >20 needs shortlist/budget.
4. `test.csv` is grouped by intent — shuffle before sampling (same trap twice avoided).
5. Trained linear models dominate fixed-label English tasks (0.927); zero-shot
   Laya (0.693) and Jev (0.793) compete on flexibility, not accuracy.
6. NLI entailment (0.533) is not "better Laya" — mask-scoring wins here.
7. Full head backprop peaks at 3.38 GB — 16 GB Macs can genuinely train Laya.
8. Short training sharpens probabilities before it flips predictions (NLL/ECE
   down, accuracy flat) — expect calibration wins first.
9. Jev assigns zero probability to gold on ~5% of rows — never branch on a
   single probability without a fallback.
10. Data residency decides the stack more than accuracy does (see §Prospects).

## Tools

| tool | does |
|---|---|
| `laya_classify.py` | `LayaClassifier` wrapper + CLI |
| `tutorial_laya.py` | live choice/noul/score demo |
| `bench_classify.py` | latency/accuracy/gating/scaling |
| `compare_runtimes.py` | torch vs MLX head-to-head |
| `demo_router.py` / `demo_llm_couple.py` | multilingual routing / gate+router with stub LLM |
| `notebooks/laya_customer_classification.ipynb` | 15-cell Banking77 walkthrough |
| `scripts/fetch_demo_data.py` / `fetch_finphrasebank.py` | vendor both datasets |
| `scripts/classify_demo_data.py` | feeds the web demo |
| `scripts/bench_comparison.py` / `bench_jev.py` | Laya+sklearn+NLI / live Jev |
| `scripts/train_laya_local.py` | `calibrate` + `head` stages |
| `scripts/verify_offline.py` / `fetch_models.py` | offline proof / re-vendor weights |
| `web/` | Next.js explorer over `results.json` |

## Capabilities, one line each

- **Laya**: local typed decisions (choice/score/noul), zero-shot via
  descriptions, one pass per doc, multilingual via Router, fine-tunable.
- **Jev**: best measured zero-shot (0.793) with zero ops — hosted, paid,
  data leaves the network.
- **sklearn**: best accuracy/latency on frozen English labels; needs labels.
- **bart-mnli**: local zero-shot fallback; slower and weaker here.
- **Small local LLM**: prose and reasoning; Laya decides when to call it.

## Prospects (BFSI, local-first)

Near term: temperature-fit on your grievance labels; head-tune with more
epochs/unfrozen top layers; extend the web explorer to PhraseBank; pilot the
gate pattern in front of any internal LLM. Medium term: full RLCD fine-tune on
Kaggle GPUs for your schema; `typed-decisions`-style workflows for ops queues;
Hindi/Hinglish routing for real complaint traffic. Structural: every hosted
component needs a residency exception — prefer the local column by default.

## Doc map

`README.md` (index) → `FINDINGS.md` (setup numbers) → `TUTORIAL.md` (how Laya
works) → notebook (hands-on) → `COMPARISON.md` (benchmarks) → `TRAINING.md`
(training) → `PRODUCTS.md` (what to build) → this file (what happened).

# jev-laya-test

Scratch workspace for evaluating **Laya** (open local typed-decision model) and **Jev**
(TypeSafe's hosted equivalent) as local classifiers. Start with **[JOURNEY.md](JOURNEY.md)**
— the full log of what was tried, in order — or jump to a doc below.

Everything needed to run this is **reproducible from what's committed**: the pinned
environment (`uv.lock` + `requirements.txt`) installs from the network, and the
**weights** (~3.2 GB) and **wheelhouse** (~221 MB) are re-fetched on demand — they
stay out of the pushed history (GitHub 100 MB/file cap). After fetching, it runs
with no network.

## Quick start

```bash
uv sync --frozen                            # exact 38-package environment from uv.lock
uv run scripts/fetch_models.py              # fetch weights into models/ (~3.2 GB, Hub)
uv run scripts/verify_offline.py            # proves requirements + weights work offline
uv run laya_classify.py "the API returns 500 on every request"
```

`uv sync --frozen` reproduces the pinned environment. To install from a local wheelhouse
instead — no network at all, and the way to rebuild this on another Mac, first build it:

```bash
uv venv --python 3.12 .venv
uv run --no-project --with pip python -m pip download -r requirements.txt -d vendor/wheels --only-binary=:all:
uv pip install --no-index --find-links vendor/wheels -r requirements.txt
```

Build the wheelhouse first: only its `SHA256SUMS` manifest is tracked, the `*.whl`
blobs stay out of git (see below).

The wheelhouse is platform-specific (**macOS 14+ arm64, CPython 3.12**); on Linux/CUDA delete the
pinned host wheels and let `uv sync` resolve fresh — `laya` is pure Python and works anywhere
`torch` does.

## What's committed (and what's fetched)

| path | size | what |
|---|---:|---|
| `uv.lock` | 73 KB | resolved dependency graph behind `uv sync --frozen` |
| `requirements.txt` | 1 KB | flat pin list (wheelhouse input, verified against the lock) |
| `vendor/wheels/SHA256SUMS` | 4 KB | manifest of the 38-wheel house (blobs excluded from git) |
| `data/`, `benchmarks/`, `notebooks/`, `web/public/results.json` | ~2 MB | vendored datasets, measured results, notebook, web demo data |
| `training/*.json` | KBs | calibration + head-tune reports (weight `.pt` files excluded, reproducible via script) |

Fetched on demand, never pushed (GitHub 100 MB/file cap):

| path | size | how |
|---|---:|---|
| `models/laya/` | 2.37 GB | `uv run scripts/fetch_models.py` — `english` (bundle root) + `multilingual/` + `typed-decisions/` |
| `models/laya-mlx/` | 846 MB | same script — the `english` checkpoint pre-converted to MLX fp16 |
| `vendor/wheels/*.whl` | 221 MB | `pip download -r requirements.txt -d vendor/wheels` (manifest stays tracked) |

## Try it

```bash
# concepts first: choice / noul / score, live
uv run tutorial_laya.py

# classify a string
uv run laya_classify.py "the API returns 500 on every request"

# notebook: 3,080 real bank queries -> 7 categories, no training
# (open notebooks/laya_customer_classification.ipynb in Jupyter/VS Code)

# head-to-head: Laya vs Jev vs TF-IDF+LogReg (all measured)
uv run --with scikit-learn scripts/bench_comparison.py   # see COMPARISON.md + TRAINING.md

# Laya + LLM coupling (gate + router, stub LLM, fully local)
uv run demo_llm_couple.py   # see PRODUCTS.md for BFSI-local use cases

# web demo of the notebook's results (accuracy bars, confusions, query explorer)
cd web && npm install && npm run dev   # http://localhost:3000

# accuracy, latency, confidence gating, and the cost of extra questions
uv run bench_classify.py

# PyTorch/MPS vs MLX on this machine
uv run compare_runtimes.py

# multilingual routing, and English-on-Hindi
uv run demo_router.py

# verify the offline setup end to end
uv run scripts/verify_offline.py --compare-runtimes
```

## Use as a library

```python
from laya_classify import LayaClassifier, LABELS

clf = LayaClassifier(LABELS)                    # or pass your own {label: description}
pred = clf.predict("I was billed twice.")       # -> Prediction(label, confidence, probabilities)
```

Pick a runtime or a checkpoint:

```python
LayaClassifier(LABELS, runtime="mlx")        # force MLX        (Apple Silicon)
LayaClassifier(LABELS, runtime="torch")      # force PyTorch    (MPS / CUDA / CPU)
LayaClassifier(LABELS, subfolder="multilingual")       # 100+ languages
LayaClassifier(LABELS, subfolder="typed-decisions")    # upstream decision workflows
```

`runtime="auto"` (the default) uses MLX when it is importable and falls back to PyTorch.

## Where the weights come from

`laya_paths.py` resolves every load to the vendored tree when it is present and otherwise to the
Hugging Face Hub, so nothing is hard-coded to this checkout:

| checkpoint | vendored path | params | Hub id when not vendored |
|---|---|---:|---|
| `english` | `models/laya` | 421M | `convaiinnovations/laya` |
| `multilingual` | `models/laya/multilingual` | 322M | `convaiinnovations/laya` + `subfolder` |
| `typed-decisions` | `models/laya/typed-decisions` | 421M | `convaiinnovations/laya` + `subfolder` |
| `english` (MLX) | `models/laya-mlx` | 421M | `aac6fef/laya-mlx` |

Both runtimes accept a plain directory and append `subfolder`, which is what makes the nesting
above work. Point `LAYA_MODELS_DIR` elsewhere to use a different tree; delete `models/` and
everything downloads from the Hub instead.

To refresh or re-vendor:

```bash
uv run scripts/fetch_models.py                      # english + multilingual + MLX export
uv run scripts/fetch_models.py --checkpoints all    # add typed-decisions
uv run scripts/fetch_models.py --force              # re-fetch
```

## Requirements notes

- **Python 3.12.** `uv` manages it; the system Python here is 3.9, which Laya will not install on.
- **Always export `USE_TF=0`.** `transformers` probes for TensorFlow at import and its abseil
  runtime can deadlock model construction. Every script here sets it before importing.
- `uv venv` ships without `pip`, so the wheelhouse was built with
  `uv run --no-project --with pip python -m pip download -r requirements.txt -d vendor/wheels --only-binary=:all:`.

## Weights and wheels are fetched, not committed

`models/` (~3.2 GB) and `vendor/wheels/*.whl` (~221 MB) stay out of the pushed
history — GitHub rejects any plain-tree file over 100 MB, and the checkpoints are
644–846 MB each. What you clone is a ~2 MB repo; you fetch the blobs with:

```bash
uv run scripts/fetch_models.py                      # english + multilingual + MLX export
uv run scripts/fetch_models.py --checkpoints all    # add typed-decisions
```

Why not LFS / release assets / external store:

| option | limit | verdict |
|---|---|---|
| Repo tree, no LFS | **100 MB per file** (hard reject) | 4 checkpoint files of 644–846 MB → rejected |
| Git LFS per file | 2 GB (Free/Pro) | would fit (largest 846 MB), but… |
| Git LFS free bandwidth | 10 GiB / month | **every fresh clone would spend ~3.0 GiB (~30%)** |
| GitHub release assets | 2 GiB per file, not LFS-billed | viable if you ever need one-click blobs |

So the repo keeps the public upstream artefacts out and re-fetches them in a few
minutes (`scripts/fetch_models.py`; `uv sync` for wheels). `git-lfs` is not installed
here and not needed. The wheelhouse's 127 MB `torch` wheel has the same problem in
miniature, hence only `vendor/wheels/SHA256SUMS` is tracked.

## Headline numbers (M1 Pro, 16 GB)

| runtime | load | p50 | p95 | docs/s | accuracy |
|---|---:|---:|---:|---:|---:|
| PyTorch (MPS) | 43.0 s | 59.9 ms | 65.6 ms | 16.8 | 0.889 |
| MLX (FP16) | 0.1 s | 42.7 ms | 44.5 ms | 23.6 | 0.889 |

Same run, same documents, loading from `models/`: **18/18 identical labels**, MLX 1.40x faster at
p50. The 0.1 s MLX load is the vendored pre-converted fp16 export; converting the bundled
checkpoint in memory instead costs ~0.8 s. Do not gate on raw confidence until you have fitted
temperatures on your own data — see FINDINGS.md §3.

# jev-laya-test

Scratch workspace for evaluating **Laya** (open local typed-decision model) and **Jev**
(TypeSafe's hosted equivalent) as local classifiers. See **[FINDINGS.md](FINDINGS.md)** for
measured results and the reasoning.

Everything needed to run this is committed: the **wheels** (`vendor/wheels/`, 221 MB) and the
**weights** (`models/`, 3.2 GB). It installs and runs with no network.

## Quick start

```bash
uv sync --frozen                            # exact 38-package environment from uv.lock
uv run scripts/verify_offline.py            # proves requirements + weights work offline
uv run laya_classify.py "the API returns 500 on every request"
```

`uv sync --frozen` reproduces the pinned environment. To install from the vendored wheelhouse
instead — no network at all, and the way to rebuild this on another Mac:

```bash
uv venv --python 3.12 .venv
uv pip install --no-index --find-links vendor/wheels -r requirements.txt
```

The wheelhouse is platform-specific (**macOS 14+ arm64, CPython 3.12**); on Linux/CUDA delete the
pinned host wheels and let `uv sync` resolve fresh — `laya` is pure Python and works anywhere
`torch` does.

## What's committed

| path | size | what |
|---|---:|---|
| `models/laya/` | 2.37 GB | upstream checkpoints: `english` (bundle root) + `multilingual/` + `typed-decisions/` |
| `models/laya-mlx/` | 846 MB | the same `english` checkpoint pre-converted to MLX fp16 |
| `vendor/wheels/` | 221 MB | all 38 wheels + `SHA256SUMS` |
| `uv.lock` | 73 KB | resolved dependency graph behind `uv sync --frozen` |

## Try it

```bash
# concepts first: choice / noul / score, live
uv run tutorial_laya.py

# classify a string
uv run laya_classify.py "the API returns 500 on every request"

# notebook: 3,080 real bank queries -> 7 categories, no training
# (open notebooks/laya_customer_classification.ipynb in Jupyter/VS Code)

# head-to-head: Laya vs TF-IDF+LogReg (measured) vs Jev (cited)
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

## Committing 3.2 GB of weights

`models/` is committed, so the repository is ~3.4 GB and a fresh clone transfers all of it. Against
GitHub's current limits:

| | limit | this repo |
|---|---|---|
| Repo tree, no LFS | **100 MB per file** (hard reject) | 4 checkpoint files of 644–846 MB → rejected |
| Git LFS per file | 2 GB (Free/Pro), 4 GB (Team) | largest is 846 MB → fits |
| Git LFS free storage | 10 GiB / month (Free/Pro) | 3.0 GiB → fits |
| Git LFS free bandwidth | 10 GiB / month (Free/Pro) | **every fresh clone downloads ~3.0 GiB** |

Two consequences worth knowing before you push:

1. **`git-lfs` is not installed on this machine** (`git lfs version` → not a git command), so the
   weight files cannot be pushed at all until you `brew install git-lfs` and `git lfs install`,
   then re-add the weights so they are tracked via LFS (`.gitattributes` with `*.safetensors`).
2. Even with LFS, each clone spends ~30% of the free monthly bandwidth, and pushing a *modified*
   weight file bills its full size again.

Cheaper alternatives, roughly in order:

- **Don't ship the weights.** They are public upstream artefacts and
  `uv run scripts/fetch_models.py` re-fetches them in a few minutes. Keeps the repo at ~230 MB.
- **GitHub release assets** — 2 GiB per file and *not* billed as LFS; attach one per checkpoint.
- **An external store** (Hugging Face, S3, or plain `rsync` of `models/`).

The wheelhouse has the same problem in miniature: it is 221 MB including a **127 MB `torch` wheel**
over the 100 MB tree limit. Either track `vendor/wheels/*.whl` with LFS too, or keep the wheels out
and let `uv sync` fetch them.

To drop the weights before pushing anything (nothing is pushed yet, so this is clean):

```bash
git reset --soft HEAD~1          # uncommit, keep the files staged
git restore --staged models/     # unstage them
echo 'models/' >> .gitignore     # keep them on disk, out of history
```

## Headline numbers (M1 Pro, 16 GB)

| runtime | load | p50 | p95 | docs/s | accuracy |
|---|---:|---:|---:|---:|---:|
| PyTorch (MPS) | 43.0 s | 59.9 ms | 65.6 ms | 16.8 | 0.889 |
| MLX (FP16) | 0.1 s | 42.7 ms | 44.5 ms | 23.6 | 0.889 |

Same run, same documents, loading from `models/`: **18/18 identical labels**, MLX 1.40x faster at
p50. The 0.1 s MLX load is the vendored pre-converted fp16 export; converting the bundled
checkpoint in memory instead costs ~0.8 s. Do not gate on raw confidence until you have fitted
temperatures on your own data — see FINDINGS.md §3.

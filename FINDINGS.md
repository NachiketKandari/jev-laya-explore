# Findings: Laya and Jev on this machine

Machine: MacBook Pro **M1 Pro, 16 GB**, macOS 26 (Darwin 25.6), Python 3.12.14 via `uv`,
`torch 2.14.0` (MPS) and `mlx 0.32.2`. Model weights: **3.2 GB, vendored in `./models`**
(all three checkpoints plus a pre-converted MLX export), with the 221 MB wheelhouse in
`vendor/wheels`. Nothing here needs the network — see §6.

## TL;DR

- **Laya runs well here and is a genuinely good local classifier**: ~40 ms per decision, no
  output tokens, no JSON to repair, 25 docs/s in a single process.
- **Use the MLX runtime, not PyTorch**, on this hardware: 42.7 ms vs 59.9 ms p50, 0.1 s vs
  43.0 s to load, identical labels on every test, same 0.889 accuracy.
- **Jev itself cannot be run locally** — it is a closed hosted API. The local option
  (`githubnext/localjev`) is a bridge that needs Bun, oMLX and a 26B model; neither tool is
  installed here and that model would not fit comfortably in 16 GB.
- **The payoff**: Laya's `system_one` payload is *the same shape* as the Jev / LocalJev
  response, so Laya can back a Jev-shaped client unchanged. See §4.
- **Do not gate on the raw confidence.** On our set, mean confidence was 0.551 when right and
  0.543 when wrong, and every threshold above 0.5 *lowered* accuracy.

## 1. What each thing actually is

| | Laya | Jev (TypeSafe) |
|---|---|---|
| What | non-autoregressive "System 1" decision model: ModernBERT-large encoder + decision heads | same idea (System One model) |
| Output | `choice` / `score` / `noul`, plus `confidence` and `action` | `choice` / `score` / `noul` |
| Weights | Apache-2.0, open, on HF (`convaiinnovations/laya`) | closed |
| Runtime | local, in-process (`pip install laya`) | hosted API, $0.042 / 1M tokens |
| Local option | `laya` (PyTorch) or `laya-mlx` (native MLX) | none — `localjev` bridges to a *different* local model |

Jev returns typed answers instead of generated text, which is why it is fast. Laya is an open
implementation of the same primitive: the repo benchmarks it at ~7.8x Jev's p50 latency (32.8 ms
vs 236–276 ms; Jev figures are third-party) and the model card is candid about where Jev still
wins — label spaces with 50+ options, and soft distribution matching.

## 2. Measured on this machine

`python compare_runtimes.py` — the same 18 labelled support tickets, one `choice` question,
one document per call, warm:

| runtime | model load | p50 | p95 | docs/s | accuracy |
|---|---:|---:|---:|---:|---:|
| PyTorch (MPS) | 43.0 s | 59.9 ms | 65.6 ms | 16.8 | 0.889 |
| MLX (FP16) | 0.1 s | 42.7 ms | 44.5 ms | 23.6 | 0.889 |

- Runtime label agreement: **18/18 identical**.
- **MLX is 1.40x faster** at p50, and loads in a tenth of a second instead of most of a minute.
  That 0.1 s is the pre-converted fp16 export; converting the bundled checkpoint in memory
  costs ~0.8 s instead.
- `python bench_classify.py` on the MLX default: p50 **40.9 ms**, p95 ~45 ms, **~24 docs/s**.
- Latencies are stable to a few ms run to run; load times and p95 are the noisier figures.
- Accuracy was 16/18 on hand-labelled tickets, misses being `sales -> billing` and
  `cancellation -> other`. For scale, the model card reports 0.950 on AG News; 0.889 here is a
  plausible 5-class intent result, not a benchmark.
- Confidence statistics were **identical across both runtimes** (0.551 right / 0.543 wrong),
  which is good evidence the MLX port is faithful, not just fast.

(The upstream README measures 13.4 ms p50 on an M3 Max and 39.5 ms on a T4. An M1 Pro GPU is
the difference — expect ~40 ms per decision here.)

### Cost of asking more questions in one call

One fixed document, varying only the number of `choice` questions:

| | MLX | PyTorch (MPS) |
|---|---:|---:|
| 1 question, one call | 40.9 ms | 55.2 ms |
| 3 questions, one call | 112.6 ms | 115.2 ms |
| 5 questions, one call | 168.4 ms | 197.0 ms |
| 5 questions, 5 separate calls | 218.4 ms | 283.2 ms |

(1 and 5 questions come from one `compare_runtimes.py` run in both columns; the MLX 3-question
and separate-call rows and the PyTorch 3-question row are from the adjacent `bench_classify.py`
and earlier runs. Jitter is a few ms.)

Batching every question into one forward pass saves **23% (MLX)** to **30% (PyTorch)** versus
separate calls, but each question row **re-encodes the state**, so marginal cost stays real:
~30 ms/question on the English checkpoint, and only **~11.5 ms** on the multilingual one
(322M params; 16.5 ms for one question, 62.4 ms for five). Batching buys latency amortisation,
not free extra labels.

Laya exposes no multi-document batch API — one document per forward pass — so single-process
throughput is ~25 docs/s. For volume, run several worker processes.

## 3. Using Laya for local classification

`laya_classify.py` wraps a `choice` question into an ordinary classifier:

```python
from laya_classify import LayaClassifier, LABELS

clf = LayaClassifier(LABELS)          # runtime="auto" -> MLX on this Mac
pred = clf.predict("I was billed twice for March.")
pred.label, pred.confidence           # 'billing', 0.959
```

The recipe, roughly in order of how much each part matters:

1. **Define labels at request time, with descriptions.** Every option is scored at its own
   `[MASK]` token, so new labels need no retraining — but `"invoices, payments, refunds"` is
   what makes a label distinguishable from `"bugs, outages"`. This is the biggest lever.
2. **Pick the runtime for the hardware.** `runtime="auto"` uses MLX where importable, else
   PyTorch. On CUDA or CPU, `import laya` is the same API.
3. **Route before you classify.** `Router(preload=["english", "multilingual"])` keeps both
   checkpoints resident (pointed at `models/` by `laya_paths.model_for()`) and picks one per
   request using script/language detection that costs microseconds. Skipping it
   is a correctness bug, not an optimisation:

   ```
   hindi ticket -> English checkpoint:      intent=refund  confidence=0.509
   hindi ticket -> multilingual checkpoint: intent=refund  confidence=0.997
   ```

   Here the English checkpoint still got the label right but lost half its confidence; the
   documented failure on unseen scripts is a *confident wrong* answer, which no threshold can
   catch. Router correctly sent Hindi (devanagari script) and German (Latin but not English)
   to the multilingual checkpoint.
4. **Ask for everything in one call.** The `triage_questions()` preset returns intent,
   urgency, frustration, refund-requested and churn-risk in ~150 ms total — five answers for
   four extra questions' worth of time, in one pass, with zero output tokens.
5. **Calibrate before gating.** See the table below.
6. **Set `USE_TF=0`.** `transformers` probes for TensorFlow at import; if TF is installed its
   abseil runtime can deadlock model construction. All scripts here set it before importing.

### Gotchas we actually hit

- On load, **both** runtimes warn: `this checkpoint ships temperatures outside [0.5, 5] which
  would distort confidence; clamping choice:11+=0.1006`. `english` and `typed-decisions` each
  ship a fitted temperature of **0.1006** for the `choice:11+` bucket — far below the [0.5, 5]
  clamp, and low enough that applying it raw would make an 11-option question sharply peakier
  than calibrated. `multilingual` ships `[1.0, 1.0, 1.0]` and no per-option-count temperatures,
  so it is unaffected. Confidence from `choice:11+` is uncalibrated on either runtime.
- **Raw confidence does not separate right from wrong:**

  | gate | traffic kept | accuracy |
  |---|---:|---:|
  | none | 100% | 0.889 |
  | >= 0.50 | 56% | 0.900 |
  | >= 0.85 | 17% | 0.667 |
  | >= 0.95 | 11% | 0.500 |

  Everything above 0.5 made accuracy worse than not gating, and 11% of traffic was kept at
  0.500 — i.e. coin-flip. The model card says the same (raw ECE 0.213, 0.081 after fitting one
  temperature per question type and option count). Fit temperatures on your own data first.
- **>20 options degrades sharply.** All option texts share a fixed `head_max_len` budget (192
  tokens on the English checkpoint), so a 77-label question gets ~3 tokens per label and
  accuracy falls to 0.425 (Banking77) versus Jev's 0.870. Use `laya.predict_shortlist(...)`
  with an embedding function to keep the top-k, or raise `agent.cfg["head_max_len"]`.
- `predict` and `system_one` are the same method; answers live at `result["answers"][qid]` with
  `choice`/`score`/`noul`, `confidence` and `probabilities`.
- MLX checkpoint resolution: `laya_mlx.load()` converts the bundled checkpoint to fp16 in
  memory (~0.8 s), while the pre-converted export starts in ~0.1 s but ships only `english`. So
  `english` wants the export and `multilingual`/`typed-decisions` want the bundle plus
  `subfolder=`. `laya_paths.model_for()` encodes exactly that rule — it is why
  `compare_runtimes.py` now reports a 0.1 s MLX load instead of 0.8 s.

## 4. Jev, concretely

Jev is a hosted, closed-weight API: you point an SDK at TypeSafe with an API key, so there is no
"install Jev locally". `githubnext/localjev` (cloned to `./localjev`) is a TypeScript/Bun server
that *emulates the Jev wire protocol* on top of a local model — by default
`diffusiongemma-26B-A4B-it-4bit` behind oMLX at `127.0.0.1:8000`. Reading its source, it is an
honest bridge: oMLX does not expose the seeded-diffusion logit primitives OpenJev relies on, so
it translates typed questions into a classification prompt, asks for JSON probabilities, and
normalises them — wire-compatible, not mathematically equivalent, with the caveat that the
probabilities are self-reported.

Verdict for this machine: **not viable now.** `bun` and `omlx` are both missing, and a 4-bit
26B MoE needs roughly 13–14 GB before macOS and KV cache — on a 16 GB M1 Pro that means
swapping. Keep `localjev` for reference, not for running.

### The useful part: the wire format is identical

LocalJev's response (from `localjev/src/server.ts` and `localjev/test/server.test.ts`):

```json
{ "model": "...",
  "answers": { "urgent": { "type": "noul", "noul": 0.95 } },
  "usage": { "input_tokens": 42, "output_tokens": 7 } }
```

Laya's real `agent.system_one(...)` output on this machine:

```json
{ "model": "laya-rl-agent",
  "answers": {
    "label": { "type": "choice", "choice": "billing", "confidence": 0.8074,
               "probabilities": { "billing": 0.9704, "technical": 0.0296 },
               "action": { "act_probability": 1.0 } },
    "urgent": { "type": "noul", "noul": 0.8542, "confidence": 0.8542 },
    "frustration": { "type": "score", "score": 1.2259,
                     "legend": { "0": "calm", "1": "annoyed", "2": "angry" },
                     "probabilities": { "0": 0.0416, "1": 0.6909, "2": 0.2675 } } },
  "usage": { "input_tokens": 133, "output_tokens": 0 } }
```

Same envelope, same `answers` keys. Laya adds an `action.act_probability` field, which a Jev
client will simply ignore. So exposing a Jev-shaped `POST /v1/systemone` backed by Laya is a
thin shim around `agent.system_one()` — no model behaviour to reimplement, fully offline, and
testable against LocalJev's own request validation.

## 5. Files here

| file | what it does |
|---|---|
| `laya_classify.py` | `LayaClassifier`: a `choice` question as a normal classifier (runtime selectable), plus a CLI |
| `laya_paths.py` | resolves a checkpoint to the vendored copy or the Hub; the one place that knows the layout |
| `bench_classify.py` | accuracy on 18 labelled tickets, latency distribution, confidence gating, question scaling |
| `compare_runtimes.py` | PyTorch/MPS vs MLX head-to-head, with runtime label agreement |
| `demo_router.py` | multilingual routing, and the English-checkpoint-on-Hindi comparison |
| `scripts/fetch_models.py` | vendors the checkpoints into `models/` (idempotent) |
| `scripts/verify_offline.py` | asserts requirements + wheelhouse + offline loading, exit 1 on any gap |
| `pyproject.toml` / `uv.lock` | the pinned environment behind `uv sync --frozen` |
| `requirements.txt` | flat pin list, also the input for the wheelhouse download |
| `models/`, `vendor/wheels/` | committed weights and wheels (§6) |
| `localjev/` | cloned reference implementation of the Jev wire protocol |

## 6. Reproducing this environment offline

The environment is pinned in two places that were checked to agree: `pyproject.toml` +
`uv.lock` for `uv sync --frozen`, and `requirements.txt` as a flat list. A bare `uv lock`
resolves 59 packages because the lock is universal (Linux/CUDA transitive deps included); on this
Mac `uv sync --frozen` installs exactly the 38 verified packages, byte-identical to the freeze
captured before locking.

Vendored artefacts:

| path | size | contents |
|---|---:|---|
| `models/laya/` | 2.37 GB | `english` at the root, `multilingual/` and `typed-decisions/` beneath it |
| `models/laya-mlx/` | 846 MB | `english` pre-converted to MLX fp16 |
| `vendor/wheels/` | 221 MB | 38 wheels + `SHA256SUMS` |

Only loader files are vendored — `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`,
`tokenizer/*` (+ `mlx_config.json` for the MLX export). The upstream repo is 2.4 GB all-in, so the
8.4 MB of assets, eval results and training scripts are skipped.

```bash
uv sync --frozen                                             # exact env from uv.lock
uv pip install --no-index --find-links vendor/wheels -r requirements.txt  # fully offline
uv run scripts/fetch_models.py [--checkpoints all] [--force]
uv run scripts/verify_offline.py [--compare-runtimes]
```

`verify_offline.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing Laya, so
an incomplete vendored tree is a hard error instead of a silent download. It currently reports:
38/38 pins installed, 38/38 wheel checksums matching, all four checkpoints load offline, and
PyTorch/MLX agreeing on the label with the network off.

### Gotchas found while vendoring

- **`snapshot_download(local_dir=X, allow_patterns=["sub/*"])` writes to `X/sub/`, not `X/`.**
  Passing the subfolder as `local_dir` *and* prefixing the patterns nests it one level too deep —
  `models/laya/multilingual/multilingual/`. Keep `local_dir` at the bundle root and let the
  prefixes place the siblings.
- **`du -sh ~/.cache/huggingface/hub/models--convaiinnovations--laya` reports 3.4 MB, not 2.4 GB.**
  The Hub's Xet cache keeps content-addressed blobs in one shared `hub/blobs/` store and only
  relative symlinks per repo, so per-repo `du` (which does not follow symlinks) misses the
  weights entirely. Ask the repo's `trees/<rev>.json` for real sizes: `files[name].size`.
- **`uv venv` ships no `pip`**, so `pip download` needs
  `uv run --no-project --python 3.12 --with pip python -m pip download ...`. `--only-binary=:all:`
  is worth keeping: it turns a missing wheel into a build failure rather than a silent sdist.
- **`snapshot_download` leaves a `.cache/huggingface` tree inside `local_dir`**; `fetch_models.py`
  deletes it so the vendored tree contains only checkpoint files.
- Unauthenticated Hub requests work but warn above; the copies were near-instant because the
  Hub reuses its local Xet blobs (APFS clones, so the vendored files are independent: link count
  1, safe to commit and modify).

### Committing weights

Git stores these as full blobs, so the repo is ~3.4 GB and every clone transfers it. GitHub's
limits (checked 2026-09): a **100 MB per-file cap** in a plain tree, **2 GB per file** via Git LFS
on Free/Pro, and **10 GiB/month** each of LFS storage and bandwidth. So the 644–846 MB checkpoint
files fit in LFS's 3.0 GiB of 10 GiB storage — but every fresh clone spends ~3.0 GiB of the 10 GiB
monthly bandwidth, and re-pushing a changed weight file bills its full size again.

`git-lfs` is also **not installed here** (`git lfs version` → not a git command), so the weights
cannot be pushed as-is; `brew install git-lfs` + `git lfs install` + a `.gitattributes` for
`*.safetensors` would be needed. The 221 MB wheelhouse has the same problem in miniature: its
127 MB `torch` wheel is over the 100 MB tree limit. See README.md for the re-fetch alternative and
the commands to uncommit `models/` (nothing has been pushed, so it is a clean `reset --soft`).

## Sources

- Laya model card: <https://huggingface.co/convaiinnovations/laya>
- Laya repo: <https://github.com/NandhaKishorM/laya> · PyPI: <https://pypi.org/project/laya/>
- laya-mlx: <https://pypi.org/project/laya-mlx/> (<https://github.com/mizorewww/laya-mlx>)
- LocalJev: <https://github.com/githubnext/localjev>

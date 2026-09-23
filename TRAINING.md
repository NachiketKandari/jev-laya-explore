# Training: Laya fine-tuning vs classic ML training

Two different things share the word "training" here. Do not confuse them.

## 1. Fine-tuning Laya (RLCD) — the canonical repo

**Repo: [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya)** (17.2k★,
1.5k forks, Apache-2.0) — the popular, upstream home of Laya. Start here, not
with forks: it holds the checkpoints, `BENCHMARKS.md`, and the fine-tuning
notebook:

- [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
  — the whole loop on Kaggle's free 2×T4: env check → install → preprocess
  1,200 cases (`LocalLLaMA/typed-decisions`) → write `train_ddp.py` → `torchrun`
  DDP fine-tune → evaluate 400 cases / 2,000 decisions → compare vs Jev → push
  to Hub → export JSON report. README quotes ~4–5 h for 4 epochs over ~30k
  questions; the notebook's own §5 says minutes for its smaller run.

**What training optimises (from the model card):** RLCD — the policy reports a
distribution, exploration adds zero-mean Gaussian noise to the logits, reward
is a strictly proper scoring rule (log + spherical, plus ranked-probability
score for `score` questions), REINFORCE with a group-mean baseline (GRPO-style),
TD(λ=1.0) over prefix slices for multi-turn. Honest probabilities are the only
way to maximise reward — that is where the calibration comes from.

**Why it matters:** base checkpoints score ~0.36 zero-shot on typed-decisions
(near chance); the fine-tuned one hits **0.766**, above Jev's published 0.727.
The lesson ports to our demo: Laya's 0.693 zero-shot is the floor, not the
ceiling — domain fine-tuning is the lever after descriptions.

**Not verified here:** I have not run the notebook (needs Kaggle GPUs). The
above is read from the repo, not executed.

## 1b. What fits on a 16 GB MacBook Pro (measured)

Yes — real training fits, with one exception. Measured on this M1 Pro via a
forward+backward probe on the vendored torch checkpoint (fp32, MPS):

| batch | weights | +forward peak | +backward peak |
|---|---|---|---|
| 1 | 1.69 GB | 1.80 GB | 3.38 GB |
| 4 | 1.69 GB | 2.11 GB | 3.38 GB |

Gradients dominate (≈1.7 GB); activations are small at these lengths. Add Adam
(m+v ≈ 3.4 GB fp32) and full supervised fine-tuning peaks ≈ 7 GB — inside
16 GB unified memory with room to spare. Ranked by what to try first:

1. **Temperature fitting (today, CPU, minutes).** No gradients at all: collect
   a few hundred labelled decisions, fit one temperature per (type, option
   count), ECE 0.2+ → ~0.08. Biggest calibration win per unit effort.
2. **Supervised fine-tune, head-first (fits, MPS).** Freeze the ModernBERT
   encoder, train only the 2-layer decision head + scorer on your
   (state, questions, targets). Tiny trainable count, batch 8–32 fine.
3. **LoRA on the encoder + head (fits, MPS).** A few M trainable params,
   AdamW, batch 4–8 at seq 512. Slow (MPS training throughput is modest) but
   workable overnight for thousands of examples.
4. **Full RLCD/GRPO (not here).** Group rollouts multiply activations by G and
   need sampling + KL machinery the pip package does not ship — the training
   code lives in the GitHub notebook, which assumes CUDA DDP. Use Kaggle 2×T4
   (free) or a GPU box; keep the Mac for 1–3 and for inference.

Caveat: the pip `laya` package ships **no training API** (verified: no
train/LoRA code in `laya/`); you assemble the loop yourself from `build_model`,
`build_sequence`, `collate_items` (all importable) or port the notebook's
`train_ddp.py`. The probe above used exactly those internals.

## 2. Training the sklearn baseline (what `bench_comparison.py` does)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50_000),
    LogisticRegression(max_iter=1000),
)
model.fit(train_texts, train_coarse_labels)  # 10,003 rows, 0.8 s on CPU
model.predict(["My top-up failed, WHY?"])    # -> topup, ~0.2 ms
```

Supervised, minutes to learn, microseconds to serve. Price: frozen label set,
English-only, one signal per model, `predict_proba` uncalibrated.

## Which to reach for

1. Better **descriptions** first (free, minutes) — they are Laya's training set.
2. **sklearn** if labels are fixed and labelled data exists (see `COMPARISON.md`).
3. **Fine-tune Laya** if the schema is yours, labels shift, or you need
   choice+score+noul with calibrated probabilities in one pass.

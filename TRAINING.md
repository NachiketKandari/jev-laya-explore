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

# Demo data

Customer queries for the notebook (`notebooks/laya_customer_classification.ipynb`)
and the web demo (`web/`).

## Source

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI, CC-BY-4.0):
13,083 real online-banking queries labelled with 77 fine intents. We use the
**test split** (3,080 rows). Upstream mirror:
`github.com/PolyAI-LDN/task-specific-datasets` (`banking_data/test.csv`).

## Files

| file | what |
|---|---|
| `banking77_test.csv` | `text,fine_label,coarse_label` — 3,080 rows |
| `banking77_train.csv` | `text,fine_label,coarse_label` — 10,003 rows (trains the sklearn baseline in `COMPARISON.md`) |
| `coarse_labels.json` | the 7 coarse groups + descriptions (doubles as Laya `criteria`) |
| `DATA_README.md` | this file |

## Why 7 coarse groups, not 77 intents

Laya scores every option in one `choice` question and all options share a
`head_max_len=192` token budget — past ~20 options each label gets ~4 tokens
and accuracy collapses (0.425 on Banking77, `FINDINGS.md` §3). The 77→7 mapping
in `scripts/fetch_demo_data.py` keeps the demo in Laya's sweet spot while still
using all 3,080 real queries. Exercise 4 in the notebook lets you feel the
77-label wall directly.

## Regenerate

```bash
uv run scripts/fetch_demo_data.py            # re-download + remap (stdlib only)
uv run scripts/fetch_demo_data.py --force    # overwrite
```

## Second dataset: Financial PhraseBank (sentiment)

[`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank):
2,264 financial-news sentences with unanimous annotator agreement, labelled
negative / neutral / positive. Split 80/20 (seed 7):

| file | what |
|---|---|
| `finphrasebank_train.csv` | `sentence,label` — 1,811 rows |
| `finphrasebank_test.csv` | `sentence,label` — 453 rows |

License **CC-BY-NC-SA-3.0** (non-commercial — fine for learning/internal eval,
not for products). Laya zero-shot with plain descriptions: **414/453 = 0.914**.
Regenerate: `uv run scripts/fetch_finphrasebank.py [--force]`.

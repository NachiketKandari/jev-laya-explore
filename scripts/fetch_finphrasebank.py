"""Fetch Financial PhraseBank (CC-BY-NC-SA-3.0, learning/eval use) for sentiment.

2,264 financial-news sentences where all annotators agreed: negative /
neutral / positive. Seeded 80/20 split, committed so evals run offline:

    data/finphrasebank_train.csv   sentence,label (1,811 rows)
    data/finphrasebank_test.csv    sentence,label (453 rows)

    uv run scripts/fetch_finphrasebank.py [--force]
"""

from __future__ import annotations

import csv
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main"
       "/data/FinancialPhraseBank-v1.0.zip")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    force = "--force" in sys.argv
    out_tr, out_te = DATA / "finphrasebank_train.csv", DATA / "finphrasebank_test.csv"
    if out_tr.exists() and out_te.exists() and not force:
        print(f"exists, skipping (use --force): {out_te}")
        return
    req = urllib.request.Request(URL, headers={"User-Agent": "jev-laya-test"})
    with urllib.request.urlopen(req, timeout=120) as r:
        zf = zipfile.ZipFile(__import__("io").BytesIO(r.read()))
    raw = zf.read("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt").decode("latin-1")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if "@" not in line:
            continue
        sent, label = line.rsplit("@", 1)
        label = label.strip().strip(".").lower()
        if label not in ("negative", "neutral", "positive"):
            continue
        rows.append((sent.strip(), label))
    assert len(rows) == 2264, len(rows)
    random.Random(7).shuffle(rows)
    cut = int(len(rows) * 0.8)
    for path, part in ((out_tr, rows[:cut]), (out_te, rows[cut:])):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sentence", "label"])
            w.writerows(part)
        print(f"wrote {path} ({len(part)} rows)")


if __name__ == "__main__":
    main()

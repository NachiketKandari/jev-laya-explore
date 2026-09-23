"""Measure Laya as a local classifier on this machine.

Three questions:
  1. How fast is it, warm, per decision?  (p50 / p95, not best case)
  2. Is it accurate, and are its confidences usable for gating?
  3. Does asking more questions cost more time?  (it should not: one pass)

    python bench_classify.py
"""

from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")

import statistics
import time

import laya
from laya_classify import LABELS, LayaClassifier

# Hand-labelled support tickets. Small on purpose: this is a smoke test of
# whether the model is usable here, not a benchmark report.
GOLD = [
    ("I was charged twice for March, please refund the duplicate.", "billing"),
    ("The API returns 500 on every request since this morning's deploy.", "technical"),
    ("Can you send pricing for the enterprise tier and a demo?", "sales"),
    ("Please cancel my subscription effective immediately.", "cancellation"),
    ("Where can I download my invoices for last quarter?", "billing"),
    ("Your iOS app crashes on launch after the update.", "technical"),
    ("Do you offer a nonprofit discount?", "sales"),
    ("I'd like to downgrade from Pro to Basic at the end of the term.", "cancellation"),
    ("My card was declined but I was still charged a late fee.", "billing"),
    ("Webhook deliveries stopped at 3am and we are missing events.", "technical"),
    ("How much does the team plan cost for 40 seats?", "sales"),
    ("I want to end the contract early, what are the terms?", "cancellation"),
    ("The login page loops back to sign-in after entering my MFA code.", "technical"),
    ("Please fix the VAT number on my account so future invoices are correct.", "billing"),
    ("Send me a quote for annual billing.", "sales"),
    ("We are moving to a competitor next month, please close the account.", "cancellation"),
    ("How do I export a CSV of all our ticket history?", "other"),
    ("Anyone around for a quick chat about your roadmap?", "other"),
]


def pct(values, p):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = min(int(round((p / 100) * (len(values) - 1))), len(values) - 1)
    return values[idx]


def section(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def main() -> None:
    clf = LayaClassifier(LABELS)
    print(f"device: {clf.device}   labels: {', '.join(clf.labels)}")

    # --- 1. warm latency ---------------------------------------------------
    section("1. Latency, warm (single document, one choice question)")
    clf.warm(3)
    samples = []
    for i in range(30):
        text = GOLD[i % len(GOLD)][0]
        samples.append(clf.predict(text).latency_ms)
    print(f"  n={len(samples)}  p50={pct(samples, 50):.1f} ms  p95={pct(samples, 95):.1f} ms  "
          f"mean={statistics.mean(samples):.1f} ms  max={max(samples):.1f} ms")
    print(f"  throughput ~{1000 / statistics.mean(samples):.1f} docs/s, single process")

    # --- 2. accuracy + confidence gating -----------------------------------
    section("2. Accuracy on 18 labelled tickets")
    correct = 0
    conf_correct, conf_wrong = [], []
    confusions = {}
    rows = []
    for text, gold in GOLD:
        pred = clf.predict(text)
        ok = pred.label == gold
        correct += ok
        (conf_correct if ok else conf_wrong).append(pred.confidence)
        if not ok:
            confusions[(gold, pred.label)] = confusions.get((gold, pred.label), 0) + 1
        rows.append((gold, pred))

    total = len(GOLD)
    print(f"  accuracy: {correct}/{total} = {correct / total:.3f}")
    if conf_correct:
        print(f"  mean confidence when right: {statistics.mean(conf_correct):.3f} (n={len(conf_correct)})")
    if conf_wrong:
        print(f"  mean confidence when wrong: {statistics.mean(conf_wrong):.3f} (n={len(conf_wrong)})")
    else:
        print("  no errors, so nothing to compare on wrong answers")

    for threshold in (0.5, 0.7, 0.85, 0.95):
        kept = [(g, p) for g, p in rows if p.confidence >= threshold]
        if not kept:
            print(f"  gate >= {threshold:.2f}: nothing kept")
            continue
        acc = sum(g == p.label for g, p in kept) / len(kept)
        print(f"  gate >= {threshold:.2f}: keeps {len(kept) / total:.0%} of traffic at {acc:.3f} accuracy "
              f"({total - len(kept)} escalated to a human)")

    for (gold, pred), n in sorted(confusions.items()):
        print(f"  confusion: {gold} -> {pred} (x{n})")

    # --- 3. how the cost scales with the number of questions --------------
    section("3. Cost of asking more questions in the same call")
    one = {"label": {"type": "choice", "instructions": "Which team?", "criteria": LABELS}}
    five = laya.triage_questions()

    # One fixed document, so the numbers differ only by question count
    # (document length changes latency more than anything else here).
    fixed_state = {
        "message": "Your app has been broken for three days and nobody replied. "
                   "If this is not fixed today we are cancelling our plan."
    }

    def timeit(questions, n=8):
        clf.warm(2)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            clf.agent.predict(fixed_state, questions)
            ts.append((time.perf_counter() - t0) * 1000)
        return statistics.mean(ts)

    def n_questions(k):
        return {
            f"q{i}": {"type": "choice", "instructions": "Which team?", "criteria": LABELS}
            for i in range(k)
        }

    t_one = timeit(n_questions(1))
    print(f"  1 question, one call        : {t_one:7.1f} ms")
    t_batched_5 = None
    for k in (3, 5):
        t = timeit(n_questions(k))
        if k == 5:
            t_batched_5 = t
        print(f"  {k} questions, one call       : {t:7.1f} ms  ({t / k:5.1f} ms/question)")
    t_separate = timeit(one) * 5
    print(f"  5 questions, 5 separate calls: {t_separate:7.1f} ms  "
          f"({t_separate / 5:5.1f} ms/question)")
    print(f"  -> batching all five into one forward pass saves "
          f"{(1 - t_batched_5 / t_separate) * 100:.0f}% versus separate calls,")
    print("     but each question row re-encodes the state, so the questions are not free:")
    print("     batching buys latency amortisation, not free extra labels.")

    # A real multi-attribute read: classify AND score AND flag in one call.
    result = clf.agent.predict(fixed_state, five)  # 5 different question types, one call
    print("\n  one pass, five answers:")
    for qid, ans in result["answers"].items():
        detail = ans.get("choice") or ans.get("score") or ans.get("noul")
        print(f"    {qid:<18} {detail!r:<20} confidence={ans['confidence']:.3f}")
    print(f"    output tokens: {result['usage']['output_tokens']}  <- nothing generated")


if __name__ == "__main__":
    main()

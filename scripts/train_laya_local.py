"""Local Laya training on a 16GB Mac: calibrate, then fine-tune the head.

Two stages (both fit -- measured fwd+bwd peak 3.38GB on MPS, fp32):

  calibrate  Capture exact choice logits on the 300-row sample, fit ONE
             temperature on half (NLL), report NLL/ECE on the other half.
             CPU/GPU minutes. No weights change.
  head       Freeze the ModernBERT encoder, supervised-train head+scorer on
             banking77_train (coarse-7) with AdamW, eval the 300-sample.
             ~10 min for 3000 rows x 2 epochs, batch 8, on MPS.

    uv run scripts/train_laya_local.py --stage calibrate
    uv run scripts/train_laya_local.py --stage head [--train-n 3000 --epochs 2]

Writes training/calibration.json and training/head_tune.json (reports only).
Head weights -> training/heads/*.pt (local, gitignored, ~100MB: reproduce
with this script instead of committing).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
import laya  # noqa: E402
from laya.common import QTYPES, build_sequence, collate_items, ece_score  # noqa: E402
from laya_paths import model_for  # noqa: E402

INSTRUCTIONS = "Which team should handle the request in `message`?"


def load_data(n_eval: int, seed: int):
    labels: dict = json.loads((ROOT / "data/coarse_labels.json").read_text())
    rows = list(csv.DictReader(open(ROOT / "data/banking77_test.csv", encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    train_rows = list(csv.DictReader(open(ROOT / "data/banking77_train.csv", encoding="utf-8")))
    return labels, rows[:n_eval], train_rows


def batch_logits(agent, texts: list[str], q: dict, batch: int = 8) -> torch.Tensor:
    """Exact pre-temperature choice logits, no grad. Returns (n, k) CPU tensor."""
    dev = agent.device
    out = []
    agent.model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            items = []
            for t in texts[i:i + batch]:
                seq, markers = build_sequence(agent.tok, {"message": t}, q,
                                              agent.cfg.get("max_len", 512),
                                              agent.cfg.get("head_max_len", 192))
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
            b = collate_items([items], agent.tok.pad_token_id)
            logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                    b["marker_pos"].to(dev), b["marker_mask"].to(dev),
                                    b["qtype"].to(dev))
            out.append(logits.float().cpu())
    return torch.cat(out)


def metrics(logits: torch.Tensor, targets: torch.Tensor, temp: float = 1.0) -> dict:
    p = torch.softmax(logits / temp, dim=-1).numpy()
    pred = p.argmax(axis=1)
    correct = (pred == targets.numpy()).astype(float)
    nll = float(-torch.log_softmax(logits / temp, dim=-1)
                [torch.arange(len(targets)), targets].mean())
    conf = p.max(axis=1)
    return {"accuracy": round(float(correct.mean()), 4),
            "nll": round(nll, 4),
            "ece": round(ece_score(conf, correct), 4)}


def stage_calibrate(agent, labels, sample) -> dict:
    names = list(labels)
    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": labels}
    texts = [r["text"] for r in sample]
    targets = torch.tensor([names.index(r["coarse_label"]) for r in sample])
    print("capturing logits for", len(texts), "rows ...", flush=True)
    logits = batch_logits(agent, texts, q)
    fit, held = logits[:150], logits[150:]
    tfit, theld = targets[:150], targets[150:]
    base = metrics(held, theld)
    best = (1.0, metrics(fit, tfit)["nll"])
    grid = [round(0.2 + i * 0.05, 2) for i in range(16)] + \
           [round(1.0 + i * 0.1, 2) for i in range(41)]
    for t in grid:
        nll = metrics(fit, tfit, float(t))["nll"]
        if nll < best[1]:
            best = (float(t), nll)
    fitted = metrics(held, theld, best[0])
    print(f"held-out baseline T=1.0: {base}")
    print(f"held-out fitted  T={best[0]}: {fitted}")
    return {"n": len(sample), "fit_n": 150, "held_n": 150,
            "baseline_T1": base, "fitted_T": best[0], "fitted": fitted,
            "note": "single temperature on choice logits; fit on half, reported on other half"}


def stage_head(agent, labels, sample, train_rows, args) -> dict:
    torch.manual_seed(args.seed)
    names = list(labels)
    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": labels}
    dev = agent.device

    for p in agent.model.encoder.parameters():
        p.requires_grad_(False)
    trainable = [p for p in agent.model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f}M / "
          f"{sum(p.numel() for p in agent.model.parameters()) / 1e6:.0f}M params")
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    pool = train_rows.copy()
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.train_n]
    eval_texts = [r["text"] for r in sample]
    eval_targets = torch.tensor([names.index(r["coarse_label"]) for r in sample])
    with torch.no_grad():
        base_logits = batch_logits(agent, eval_texts, q)
    base = metrics(base_logits, eval_targets)
    print(f"eval baseline (frozen): {base}", flush=True)

    def make_batch(rows):
        items, tg = [], []
        for r in rows:
            seq, markers = build_sequence(agent.tok, {"message": r["text"]}, q,
                                          agent.cfg.get("max_len", 512),
                                          agent.cfg.get("head_max_len", 192))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"]})
            tg.append(names.index(r["coarse_label"]))
        b = collate_items([items], agent.tok.pad_token_id)
        return b, torch.tensor(tg)

    agent.model.train()
    step, curve, t0 = 0, [], time.perf_counter()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(pool)
        for i in range(0, len(pool), args.batch):
            b, tg = make_batch(pool[i:i + args.batch])
            logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                    b["marker_pos"].to(dev), b["marker_mask"].to(dev),
                                    b["qtype"].to(dev))
            loss = loss_fn(logits, tg.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step} loss={loss.item():.4f}", flush=True)
            if step % 250 == 0:
                agent.model.eval()
                with torch.no_grad():
                    ev = metrics(batch_logits(agent, eval_texts, q), eval_targets)
                agent.model.train()
                curve.append({"step": step, "loss": round(loss.item(), 4), **ev})
                print(f"  eval@{step}: {ev}", flush=True)
    agent.model.eval()
    with torch.no_grad():
        final = metrics(batch_logits(agent, eval_texts, q), eval_targets)
    dt = time.perf_counter() - t0
    print(f"done in {dt / 60:.1f} min: {base} -> {final}")

    heads_dir = ROOT / "training" / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    wpath = heads_dir / f"head_bfsi7_e{args.epochs}_n{args.train_n}.pt"
    torch.save({k: v.cpu() for k, v in agent.model.state_dict().items()
                if not k.startswith("encoder.")}, wpath)
    return {"baseline": base, "final": final, "curve": curve,
            "epochs": args.epochs, "train_n": args.train_n, "batch": args.batch,
            "lr": args.lr, "minutes": round(dt / 60, 1),
            "weights": str(wpath.relative_to(ROOT)) + " (local only, gitignored)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calibrate", "head"], required=True)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    labels, sample, train_rows = load_data(args.n_eval, args.seed)
    agent = laya.load(*model_for())
    print(f"device={agent.device}", flush=True)
    (ROOT / "training").mkdir(exist_ok=True)

    if args.stage == "calibrate":
        rep = stage_calibrate(agent, labels, sample)
        (ROOT / "training" / "calibration.json").write_text(json.dumps(rep, indent=2) + "\n")
    else:
        rep = stage_head(agent, labels, sample, train_rows, args)
        (ROOT / "training" / "head_tune.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("wrote training/*.json")


if __name__ == "__main__":
    main()

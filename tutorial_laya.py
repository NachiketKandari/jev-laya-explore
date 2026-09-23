"""Live companion to TUTORIAL.md: every question type, one forward pass.

    uv run tutorial_laya.py

Loads the vendored english checkpoint (MLX on Apple Silicon, else PyTorch)
and prints real choice / noul / score answers plus a batched call.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("USE_TF", "0")


def main() -> None:
    from laya_paths import describe, model_for

    print(describe())

    try:
        import laya_mlx as laya

        agent, runtime = laya.load(*model_for(mlx=True)), "mlx"
    except ImportError:
        import laya  # type: ignore

        agent, runtime = laya.load(*model_for()), "torch"
    print(f"runtime: {runtime}  device={getattr(agent, 'device', 'n/a')}")

    state = {
        "message": "Your app has been broken for three days and nobody replied. "
        "If this is not fixed today we are cancelling our plan."
    }

    # --- 1. choice: classify -------------------------------------------------
    choice_q = {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned or a duplicate charge reversed",
                "technical_help": "a bug, outage or integration problem",
                "cancellation": "wants to cancel or downgrade",
                "other": "none of the other options fits",
            },
        }
    }
    ans = agent.system_one(state, choice_q)["answers"]["intent"]
    print(f"\n[choice] {ans['choice']}  conf={ans['confidence']:.3f}  {ans['probabilities']}")

    # --- 2. noul: boolean flag ----------------------------------------------
    noul_q = {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `message` communicate time pressure or a deadline?",
        }
    }
    ans = agent.system_one(state, noul_q)["answers"]["is_urgent"]
    print(f"[noul]   P(true)={ans['noul']:.3f}  conf={ans['confidence']:.3f}  "
          f"-> {'URGENT' if ans['noul'] > 0.5 else 'not urgent'}")

    # --- 3. score: ordinal level ---------------------------------------------
    score_q = {
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound in `message`?",
            "criteria": ["calm and neutral", "concerned but civil",
                         "clearly annoyed", "very angry"],
        }
    }
    ans = agent.system_one(state, score_q)["answers"]["frustration"]
    print(f"[score]  mean={ans['score']:.2f}  conf={ans['confidence']:.3f}  "
          f"{ans['probabilities']}  legend={ans['legend']}")

    # --- 4. all three at once, one forward pass -------------------------------
    batched = {**choice_q, **noul_q, **score_q}
    result = agent.system_one(state, batched)
    print("\n[batched] one call, three answers, "
          f"output_tokens={result['usage']['output_tokens']}:")
    for qid, a in result["answers"].items():
        detail = a.get("choice", a.get("score", a.get("noul")))
        print(f"  {qid:<12} {str(detail):<14} conf={a['confidence']:.3f}")

    # --- 5. preset: five production questions, still one call -----------------
    preset = laya.triage_questions()
    result = agent.system_one(state, preset)
    print(f"\n[preset] triage_questions() -> {sorted(preset)}:")
    print(json.dumps(
        {k: {kk: v[kk] for kk in ("type",) if kk in v} | {"answer": v.get("choice", v.get("score", v.get("noul")))}
         for k, v in result["answers"].items()}, indent=2))


if __name__ == "__main__":
    main()

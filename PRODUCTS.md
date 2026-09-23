# Laya + small LLM products, BFSI-local-first

Constraint first: customer financial data cannot leave the building. That rules
out hosted APIs for anything sensitive and selects the whole stack below. What
is cleared local in this repo: Laya (vendored weights, verified offline with
`scripts/verify_offline.py`), sklearn, bart-large-mnli, and any small MLX LLM
on the Mac. What is not: **Jev** — it is a hosted API at
[typesafe.ai](https://typesafe.ai/) ($42/B input tokens, early access), so
queries leave your network. Use Jev, if at all, only for non-sensitive traffic
or as the reference wire protocol (`localjev/`, `TUTORIAL.md` §5 shape-match).

## BFSI use cases that fit Laya's primitives

| use case | questions | why Laya, not an LLM call |
|---|---|---|
| grievance triage (deposit/loan/UPI/cards/fraud) | `choice` category + `noul` regulatory-escalation + `score` severity | one ~150 ms pass, no PII in a prompt log |
| phishing / fraud-email flagging | `email_questions()` (`is_phishing`, `is_spam`, `urgency`) | deterministic, auditable scores per email |
| KYC / document completeness | `choice` doc-type + `noul` missing-field flags | schema changes = new criteria, no retrain |
| alert prioritisation (fraud ops queue) | `score` risk + `noul` needs-human | calibrated ordering of analyst queues |
| audit sampling | `noul` anomaly flags over transactions-as-JSON | state can be JSON, not just text |
| PII gate before any LLM | `guard_questions()` (`sensitive_data`) | PII never reaches a generative model |
| Hindi/Hinglish complaints | `Router` → multilingual checkpoint | measured: routing beats raw confidence |
| complaint auto-drafting | Laya decides, small LLM writes the reply | decisions auditable, prose disposable |

## Three coupling patterns (Laya decides, LLM talks)

**1. Gate.** Laya screens every prompt before *and* after the LLM: jailbreak /
PII in, harmful content out. Blocked items never reach generation.

**2. Router.** `router_questions()` (difficulty/domain/needs_tools) decides:
template answer, small local LLM, or human. The LLM runs only when needed —
this is where the cost/latency win is, since local LLM tokens are the slow part.

**3. Structurer.** Laya returns typed facts (`choice` + probabilities); the LLM
receives *only those facts* and drafts prose. Auditors check the JSON, not the
paragraph. PII stays in Laya's state, out of the LLM context where possible.

`demo_llm_couple.py` runs patterns 1+2 live with a stub LLM (clearly marked —
plug in your local MLX model where indicated). No data leaves the machine.

## Making Laya better, in order

1. **Descriptions** — the training set you write, not collect (notebook ex. 2).
2. **Temperature fitting** — CPU, minutes, ECE → ~0.08 (`TRAINING.md` §1b).
3. **Router + preload** — free multilingual accuracy (`demo_router.py`).
4. **Shortlist / `head_max_len`** past ~20 options (`TUTORIAL.md` §8).
5. **Supervised head-tune → LoRA** on your labels — measured to fit in 16 GB.
6. **Full RLCD fine-tune** on GPUs for your schema — upstream 0.36 → 0.766.

## What could beat Laya (measured verdicts)

| alternative | verdict on our 300-query sample |
|---|---|
| TF-IDF + LogReg (trained) | beats it: 0.927, µs/doc — for frozen labels |
| bart-large-mnli zero-shot (407M) | loses: 0.533 at 122 ms/doc (7 passes) |
| Jev (hosted) | cited 0.870 on harder 77-label task; disqualified on residency |
| LLM-as-judge / guard LLMs | 8B+ generative, slower, GPU-hungry; better prose, worse decisions-per-ms |
| Laya fine-tuned | the actual favourite: upstream precedent says it closes the gap |

Details: `COMPARISON.md`. Training paths: `TRAINING.md`.

# Laya tutorial: typed decisions in one forward pass

Laya is not a chatbot. You give it a **state** + **typed questions**, it scores
every option at its own `[MASK]` slot in **one forward pass** and returns
**typed answers with probabilities**. No tokens generated (`output_tokens: 0`),
nothing to parse.

```python
import os
os.environ.setdefault("USE_TF", "0")  # always first: stops transformers TF probe deadlock

from laya_paths import model_for
import laya_mlx  # Apple Silicon, fast. On CUDA/CPU: `import laya` — same API

agent = laya_mlx.load(*model_for(mlx=True))
result = agent.system_one(state, questions)  # `predict` is the same method
```

`model_for()` in `laya_paths.py:53` resolves to `models/` when vendored,
Hub otherwise. `runtime="auto"` in `laya_classify.py:40` is the same idea:
MLX when importable, else PyTorch.

## 1. State: str, dict, or list

Anything JSON-ish. Dict keys are usually field names the instructions refer to
with backticks.

```python
state_str  = "I was billed twice for March, refund please."
state_dict = {"message": "I was billed twice for March, refund please."}
state_mail = {"subject": "double charge", "body": "...", "from": "a@x.com"}
```

Instructions name the field: `"Which team should handle the request in \`message\`?"`.
Detection/routing reads only values, not keys (`lang.py:106`).

## 2. `choice`: pick one label (classification)

```python
questions = {
    "label": {
        "type": "choice",
        "instructions": "Which team should handle the request in `message`?",
        "criteria": {
            "billing":   "invoices, payments, refunds, duplicate charges",
            "technical": "bugs, outages, crashes, API errors",
            "sales":     "pricing, plans, quotes, demos",
            "cancellation": "wants to cancel, downgrade, not renew",
            "other":     "none of the other options fits",
        },
    }
}
result = agent.system_one({"message": text}, questions)
ans = result["answers"]["label"]
# {'type': 'choice', 'choice': 'billing',
#  'probabilities': {'billing': 0.997, ...},
#  'confidence': 0.986, 'action': {'act_probability': 1.0}}
```

Rules:

- `criteria` as **dict is the training set**. `"billing"` alone works;
  `"billing: invoices, payments, refunds"` is what separates it from sales.
  List form (`["billing", ...]`) is allowed, means "no descriptions".
- Each option is scored at its own `[MASK]` (`common.py:58`). New labels need
  no retraining — this is the whole point.
- Budget: all options share `head_max_len=192` tokens. Up to ~20 options is
  fine; 77 labels ≈ 3 tokens each and accuracy collapses (0.425 on Banking77).
  Beyond 20, shortlist first (§8).
- `confidence` is `1 - H(p)/log(k)` (`common.py:210`): peaked distribution =
  high confidence. It does **not** separate right from wrong until calibrated
  (§9).

`laya_classify.py:96` wraps exactly this into `LayaClassifier(LABELS).predict(text)`.

## 3. `noul`: yes/no flag (probability the statement holds)

`noul` = "negation or underlap"? Think **boolean with a probability**.

```python
questions = {
    "is_urgent": {
        "type": "noul",
        "instructions": "Does `message` communicate time pressure or a deadline?",
    },
    # optional: pin what true/false mean
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing attempt?",
        "criteria": {"true": "phishing, scam, or fraud",
                     "false": "a legitimate email"},
    },
}
ans = agent.system_one(state, questions)["answers"]["is_urgent"]
# {'type': 'noul', 'noul': 0.7844, 'confidence': 0.7844, ...}
```

- `noul` is P(true). `confidence` is `max(p, 1-p)` (`agent.py:357`): confident-no
  and confident-yes both read high.
- Internally always 2 options `[false, true]` (`common.py:33`).
- Use for: urgent, spam, needs_reply, churn_risk, jailbreak, prompt_injection.
- Gate on `noul`, not `confidence`: `if ans["noul"] > 0.7: escalate`.

## 4. `score`: ordinal level (expected value over levels)

```python
questions = {
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound in `message`?",
        "criteria": ["calm and neutral", "concerned but civil",
                     "clearly annoyed", "very angry"],
    }
}
ans = agent.system_one(state, questions)["answers"]["frustration"]
# {'type': 'score', 'score': 1.9364,
#  'legend': {'0': 'calm and neutral', ...},
#  'probabilities': {'0': 0.02, '1': 0.32, '2': 0.35, '3': 0.30},
#  'confidence': 0.1515, ...}
```

- `criteria` is an **ordered list**, low → high. `score` is the expectation
  `sum(i * p_i)` (`agent.py:347`): 1.93 ≈ "between annoyed and very angry".
- `legend` echoes level → text. `probabilities` keys are `"0".."k-1"` strings.
- `confidence` is spread-based: flat across levels = low even when the mean is
  informative. Read the distribution, not just `.score`.
- Use for: frustration, urgency, harm_severity, difficulty. Anywhere an LLM
  would answer "rate 0-3".

## 5. All three in one call (the actual pattern)

One state, mixed types, one forward pass:

```python
from laya import triage_questions  # or laya_mlx — same

state = {"message": "Your app has been broken for three days... we are cancelling."}
result = agent.system_one(state, triage_questions())
# triage = intent(choice) + is_urgent(noul) + frustration(score)
#        + refund_requested(noul) + churn_risk(noul)
```

Real output shape on this Mac (MLX, english checkpoint; exact numbers move
with criteria wording — `uv run tutorial_laya.py` prints yours):

```json
{"intent": {"type": "choice", "choice": "cancellation",
            "probabilities": {"refund": 0.13, "technical_help": 0.42,
                              "cancellation": 0.44, "other": 0.01}},
 "is_urgent": {"type": "noul", "noul": 0.7844},
 "frustration": {"type": "score", "score": 1.9364,
                 "probabilities": {"0": 0.02, "1": 0.32, "2": 0.35, "3": 0.30}}}
```

Cost: 1 question ~41ms, 5 questions ~168ms on MLX — batching saves ~23%
versus separate calls but each question re-encodes state, so questions are
amortised, not free (`FINDINGS.md:62`).

## 6. Presets: copy, don't invent

| import | questions | state helper |
|---|---|---|
| `triage_questions()` | intent, is_urgent, frustration, refund_requested, churn_risk | `{"message": ...}` |
| `email_questions(cats)` | category, is_spam, is_phishing, urgency, needs_reply | `email_state(subj, body)` + `clean_email_body()` strips quotes/signatures |
| `guard_questions()` | jailbreak, prompt_injection, sensitive_data, harm_severity, topic | `{"prompt": ...}` |
| `moderation_questions()` | toxic, harassment, threat, spam, severity | `{"post": ...}` |
| `router_questions()` | difficulty, domain, needs_tools, is_sensitive | `{"request": ...}` |

```python
from laya import email_state, email_questions
state = email_state("double charge", raw_body, sender="a@x.com")
result = agent.system_one(state, email_questions())
```

`presets.py:5` is 187 lines — read it once, it is the best spec of what good
instructions/criteria look like.

## 7. Router: english vs multilingual

Checkpoints: `english` (421M, best on English), `multilingual` (322M, 100+
langs), `typed-decisions` (opt-in workflows). English collapses off-script
(Hindi 0.100 @20 options) while staying confident — so route **before**
classifying.

```python
from laya_mlx import Router
from laya_paths import model_for

router = Router(models={
    "english":      model_for(mlx=True),
    "multilingual": model_for("multilingual", mlx=True),
})
router.preload(["english", "multilingual"])  # cold load costs seconds; detection is µs

decision = router.route(state, questions)  # no forward pass
# RouteDecision(model='multilingual', reason='non-Latin script (devanagari, 78% ...)')

result = router.predict(state, questions)  # route + system_one; adds result["routing"]
result = router.predict(hindi, questions, model="english")  # force for comparison
router.unload()
```

Precedence: explicit `model=` > `task=` > `lang=` > detected script/language
(`router.py:257`). `demo_router.py:46` is the runnable version.

## 8. More than ~20 options: shortlist first

```python
from laya import predict_shortlist, embed_fn_from_agent

embed_fn = embed_fn_from_agent(agent)  # reuse loaded encoder; or pass your bi-encoder
result = predict_shortlist(agent, state, big_question, embed_fn, k=20)
# result["shortlist"][qid] = {labels, scores, k, n, passthrough}
# probabilities are over the kept 20 only
```

Without this, option texts get truncated to ~4 tokens each (`common.py:71`).

## 9. Confidence, temperature, `action`

- Raw confidence is **uncalibrated** (ECE 0.213 → 0.081 after fitting one
  temperature per type/option-count). On our 18 tickets: 0.551 right vs 0.543
  wrong; every gate above 0.5 hurt accuracy. Fit temperatures on your data
  before gating.
- `english` ships `choice:11+ = 0.1006`, clamped to `[0.5, 5]` at load
  (`common.py:228`) — expect a warning; treat 11+ option confidence as
  uncalibrated.
- `answer["action"]["act_probability"]` is a second head (escalate-to-human
  signal). Cheap second gate alongside confidence.

## 10. Runnable map

```bash
uv run tutorial_laya.py    # this file's concepts, live: choice/noul/score + batch + presets
uv run laya_classify.py "refund my double charge"  # single-label wrapper
uv run bench_classify.py   # latency / gating / scaling numbers
uv run demo_router.py      # routing across english/hindi/german
```

See `FINDINGS.md:86` for the full recipe in priority order.

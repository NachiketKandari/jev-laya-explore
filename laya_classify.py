"""Superfast local text classification with Laya.

Laya is a non-autoregressive "System 1" decision model: a ModernBERT-large
encoder plus decision heads. You hand it a *state* (text, JSON, email) and
*typed questions* (`choice`, `score`, `noul`), and it answers every question in
ONE forward pass -- no token-by-token decoding, no JSON to parse, nothing to
hallucinate. A `choice` question over a document is just a classifier.

This module wraps that into a scikit-learn-shaped API so classification code
reads normally.

    clf = LayaClassifier(LABELS)
    clf.classify("I was charged twice for March.")   # -> billing, 0.94

Weights are vendored under `./models`, so the default constructor needs no network;
see `laya_paths.py` for the layout and how to override it.

Run as a CLI:

    python laya_classify.py "the API returns 500 on every request"
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Dict, Iterable, List, Sequence, Union

# transformers probes for TensorFlow at import; when TF is installed its abseil
# runtime can deadlock model construction. Must be set before importing laya.
os.environ.setdefault("USE_TF", "0")

import laya  # noqa: E402

from laya_paths import model_for  # noqa: E402


def _pick_runtime(runtime: str) -> str:
    """Resolve `"auto"` to a runtime that is actually installed here."""
    if runtime == "torch":
        return "torch"
    if find_spec("laya_mlx") is not None:
        return "mlx"
    if runtime == "mlx":
        raise ImportError("runtime='mlx' requires laya-mlx; use runtime='torch' instead")
    return "torch"


def _load(model: str | None, device: str | None, subfolder: str | None, runtime: str):
    """Load a checkpoint, returning `(agent, runtime_used)`.

    `auto` prefers MLX, which is ~1.4x faster and loads ~50x faster than PyTorch on Apple
    Silicon. `model=None` means "the vendored copy if present, else the Hub"; pass an
    explicit repo id or path to override. MLX converts the bundled checkpoints to fp16 on
    load (~1 s); `models/laya-mlx` is a pre-converted standalone export of english.
    """
    runtime = _pick_runtime(runtime)
    path, sub = (model, subfolder) if model is not None else model_for(subfolder, mlx=(runtime == "mlx"))
    if runtime == "mlx":
        import laya_mlx

        return laya_mlx.load(path, device=device, subfolder=sub), "mlx"
    return laya.load(path, device=device, subfolder=sub), "torch"

# Support-intent labels: a name plus the evidence that makes it true. The
# descriptions are the entire "training set" -- Laya scores each option text at
# its own [MASK] token, so the label space is defined at request time and new
# labels need no retraining.
LABELS: Dict[str, str] = {
    "billing": "invoices, payments, refunds, duplicate or unexpected charges",
    "technical": "bugs, outages, crashes, broken integrations or API errors",
    "sales": "pricing, plans, quotes, demos, procurement questions",
    "cancellation": "wants to cancel, downgrade, or not renew",
    "other": "none of the other options fits",
}

INSTRUCTIONS = "Which team should handle the request in `message`?"


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    # Probability the model's own act/escalate head wants a human in the loop.
    # Cheap second signal: gate on it alongside `confidence`.
    act_probability: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"Prediction({self.label!r}, confidence={self.confidence:.3f}, {self.latency_ms:.1f} ms)"


class LayaClassifier:
    """A local classifier: one `choice` question asked over each document.

    Args:
        labels: Either a list of label names, or a mapping of name -> description
            where the description says what evidence makes that label true.
            Descriptions measurably help; use them.
        instructions: The question the model answers.
        question: Key the answer comes back under.
        model: HF repo id or local path. `None` (the default) uses the vendored copy
            under `./models` when present and otherwise downloads from the Hub.
        device: "mps", "cuda", "cpu", or None to auto-detect (picks MPS on Apple
            Silicon, CUDA if present, else CPU). Ignored by the MLX runtime.
        subfolder: Select a checkpoint out of the bundled repo, e.g.
            "multilingual" or "typed-decisions".
        runtime: "auto" (default, MLX when importable), "mlx", or "torch".
    """

    def __init__(
        self,
        labels: Union[Sequence[str], Dict[str, str]],
        instructions: str = INSTRUCTIONS,
        question: str = "label",
        model: str | None = None,
        device: str | None = None,
        subfolder: str | None = None,
        runtime: str = "auto",
    ) -> None:
        if isinstance(labels, dict):
            self.criteria = dict(labels)
        else:
            self.criteria = {name: name for name in labels}
        self.labels = list(self.criteria)
        self.instructions = instructions
        self.question = question
        self.agent, self.runtime = _load(model, device, subfolder, runtime)
        self._questions = {
            question: {
                "type": "choice",
                "instructions": instructions,
                "criteria": self.criteria,
            }
        }

    # -- core ---------------------------------------------------------------

    def predict(self, text: str) -> Prediction:
        """Classify one document. One forward pass, ~tens of ms."""
        start = time.perf_counter()
        result = self.agent.predict({"message": text}, self._questions)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        answer = result["answers"][self.question]
        return Prediction(
            label=answer["choice"],
            confidence=float(answer["confidence"]),
            probabilities={k: float(v) for k, v in answer["probabilities"].items()},
            latency_ms=elapsed_ms,
            act_probability=float(answer["action"]["act_probability"]),
        )

    def classify(self, text: str) -> str:
        """Just the winning label."""
        return self.predict(text).label

    def classify_many(self, texts: Iterable[str]) -> List[Prediction]:
        """Classify documents one at a time.

        Laya batches *questions* within a call, not documents, so this is a loop.
        Use it for offline labelling; for a server, reuse one warm agent.
        """
        return [self.predict(text) for text in texts]

    # -- convenience --------------------------------------------------------

    @property
    def device(self) -> str:
        return str(self.agent.device)

    def warm(self, n: int = 3) -> None:
        """Run throwaway calls so the first real measurement isn't the slow one."""
        for _ in range(n):
            self.predict("warm up")


def main() -> None:
    import sys

    texts = sys.argv[1:] or ["I was billed twice for March, please refund the duplicate."]
    clf = LayaClassifier(LABELS)
    for text in texts:
        pred = clf.predict(text)
        probs = "  ".join(f"{k}={v:.3f}" for k, v in sorted(pred.probabilities.items()))
        print(f"{text}\n  -> {pred.label}  conf={pred.confidence:.3f}  ({pred.latency_ms:.1f} ms)")
        print(f"     {probs}\n")


if __name__ == "__main__":
    main()

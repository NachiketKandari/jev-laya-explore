#!/usr/bin/env python3
"""Prove this project is complete *and* offline-capable.

Three checks, in increasing cost:

  1. every pin in requirements.txt is actually installed, at that version;
  2. every wheel in vendor/wheels matches the SHA-256 recorded in SHA256SUMS, and every
     requirement is covered by a wheel -- i.e. the environment can be rebuilt with no
     network at all;
  3. every vendored checkpoint loads and answers with the Hub switched off.

Check 3 sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE before importing laya, so a missing
vendored file fails loudly instead of silently downloading. That is the only way to catch
an incomplete vendored tree.

    uv run scripts/verify_offline.py                    # MLX runtime, ~10 s
    uv run scripts/verify_offline.py --compare-runtimes  # also PyTorch, ~+45 s
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import sys
import time
from pathlib import Path

# Set before anything can import huggingface_hub/transformers. A vendored tree that is
# missing a file must error here rather than quietly fetch it.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya_paths import MODELS_DIR, bundled_checkpoint, describe, model_for  # noqa: E402

CHECKPOINTS = ("english", "multilingual", "typed-decisions")
WHEELS_DIR = ROOT / "vendor" / "wheels"
REQUIREMENTS = ROOT / "requirements.txt"


def canonical(name: str) -> str:
    """PEP 503 name normalisation, so `laya-mlx` == `laya_mlx`."""
    return re.sub(r"[-_.]+", "-", name).lower()


class Report:
    """Prints PASS/FAIL lines and remembers what broke."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<46}{detail}")
        if not ok:
            self.failures.append(label)
        return ok

    def note(self, label: str, detail: str = "") -> None:
        print(f"  ..    {label:<46}{detail}")


def install_metadata() -> dict[str, str]:
    from importlib.metadata import distributions

    found = {}
    for dist in distributions():
        name = (dist.metadata or {}).get("Name")
        if name:
            found[canonical(name)] = dist.version
    return found


def read_requirements() -> list[tuple[str, str]]:
    pins = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, version = line.partition("==")
        pins.append((name.strip(), version.strip()))
    return pins


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_requirements(report: Report, pins: list[tuple[str, str]]) -> None:
    print("\n1. Installed requirements")
    installed = install_metadata()
    missing = [name for name, _ in pins if canonical(name) not in installed]
    mismatched = [
        f"{name}: want {want}, have {installed[canonical(name)]}"
        for name, want in pins
        if canonical(name) in installed and installed[canonical(name)] != want
    ]
    report.check(
        not missing and not mismatched,
        f"requirements.txt: {len(pins)} pinned packages",
        "all present at the pinned version"
        if not (missing or mismatched)
        else "; ".join(missing + mismatched),
    )
    extra = sorted(set(installed) - {canonical(name) for name, _ in pins})
    if extra:
        report.note("not pinned in requirements.txt", ", ".join(extra))


def check_wheelhouse(report: Report, pins: list[tuple[str, str]]) -> None:
    print("\n2. Offline wheelhouse (vendor/wheels)")
    manifest = WHEELS_DIR / "SHA256SUMS"
    if not manifest.is_file():
        report.check(False, "SHA256SUMS present", "run: pip download -r requirements.txt -d vendor/wheels")
        return

    entries = {}
    for line in manifest.read_text().splitlines():
        digest, _, name = line.partition("  ")
        if digest.strip() and name.strip():
            entries[name.strip()] = digest.strip()

    problems = []
    for name, digest in entries.items():
        wheel = WHEELS_DIR / name
        if not wheel.is_file():
            problems.append(f"{name}: missing")
        elif sha256(wheel) != digest:
            problems.append(f"{name}: sha256 mismatch")
    report.check(
        not problems,
        f"{len(entries)} wheels verified against SHA-256",
        "all match" if not problems else "; ".join(problems[:3]),
    )

    have = {canonical(name.split("-")[0]) for name in entries if name.endswith(".whl")}
    uncovered = [name for name, _ in pins if canonical(name) not in have]
    report.check(
        not uncovered,
        "a wheel exists for every requirement",
        "offline install is possible" if not uncovered else ", ".join(uncovered),
    )


def check_models(report: Report) -> None:
    print("\n3. Vendored checkpoints")
    for name in CHECKPOINTS:
        subfolder = None if name == "english" else name
        path = bundled_checkpoint(subfolder)
        size = ""
        if path is not None:
            size = f"{(path / 'model.safetensors').stat().st_size / 1e9:.2f} GB"
        report.check(path is not None, f"models: {name}", size or f"missing under {MODELS_DIR}")
    mlx_path = model_for(mlx=True)[0]
    report.check(
        Path(mlx_path).is_dir(),
        "models: pre-converted MLX export",
        str(Path(mlx_path).relative_to(MODELS_DIR)) if Path(mlx_path).is_dir() else mlx_path,
    )


def check_offline_load(report: Report, compare_runtimes: bool) -> None:
    print("\n4. Loading and predicting with HF_HUB_OFFLINE=1")
    import laya_mlx
    from laya_mlx import triage_questions

    questions = triage_questions()
    state = {
        "message": "I was charged twice for March and nobody has replied in three days. "
                   "Refund the duplicate charge or we are cancelling."
    }

    mlx_intent = None
    for name in CHECKPOINTS:
        subfolder = None if name == "english" else name
        model, sub = model_for(subfolder, mlx=True)
        start = time.perf_counter()
        agent = laya_mlx.load(model, subfolder=sub)
        load_s = time.perf_counter() - start

        location = Path(agent.model_dir).resolve()
        report.check(
            MODELS_DIR.resolve() in location.parents,
            f"{name}: loaded offline",
            f"{load_s:5.1f} s  {MODELS_DIR.name}/{location.relative_to(MODELS_DIR.resolve())}",
        )

        result = agent.predict(state, questions)
        answers = result["answers"]
        intent = answers["intent"]["choice"]
        if name == "english":
            mlx_intent = intent
        report.check(
            intent in questions["intent"]["criteria"],
            f"{name}: answered offline",
            f"intent={intent}  urgency={answers['is_urgent']['noul']:.2f}",
        )

    if not compare_runtimes:
        report.note("PyTorch runtime not exercised", "pass --compare-runtimes to include it")
        return

    import laya

    model, sub = model_for(None, mlx=False)
    start = time.perf_counter()
    torch_agent = laya.load(model, subfolder=sub)
    load_s = time.perf_counter() - start
    torch_intent = torch_agent.predict(state, questions)["answers"]["intent"]["choice"]
    report.check(
        torch_intent == mlx_intent,
        "PyTorch and MLX agree on the label",
        f"torch={torch_intent} mlx={mlx_intent}  ({load_s:.1f} s load)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--compare-runtimes",
        action="store_true",
        help="also load the english checkpoint with PyTorch and check both runtimes agree",
    )
    args = parser.parse_args(argv)

    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"{describe()}")
    print(f"network: disabled for the Hub (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)")

    report = Report()
    pins = read_requirements()
    check_requirements(report, pins)
    check_wheelhouse(report, pins)
    check_models(report)
    check_offline_load(report, args.compare_runtimes)

    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} check(s): " + "; ".join(report.failures))
        return 1
    print("All checks passed: requirements installed, wheelhouse intact, vendored weights load offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

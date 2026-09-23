#!/usr/bin/env python3
"""Vendor the Laya checkpoints into ./models so this project runs with no network.

Both runtimes accept a plain local directory as their model argument
(`laya.agent.Agent` and `laya_mlx.agent.Agent` test `os.path.exists()` before touching the
Hub) and then append `subfolder` to it for the bundled siblings. So one vendored
`models/laya` serves all three checkpoints:

    laya.load("models/laya")                                # english (repo root)
    laya.load("models/laya", subfolder="multilingual")      # multilingual
    laya.load("models/laya", subfolder="typed-decisions")   # typed-decisions

Only the files a loader actually opens are fetched. The upstream repo's README, benchmark
PNGs, eval results and training scripts are skipped, which is the difference between
2.4 GB of repository and 2.37 GB of checkpoints.

Usage:
    uv run scripts/fetch_models.py                      # english + multilingual + MLX
    uv run scripts/fetch_models.py --checkpoints all    # add typed-decisions
    uv run scripts/fetch_models.py --force              # re-fetch even if present
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# convaiinnovations/laya bundles all three checkpoints, with english at the repo root.
BUNDLE_REPO = "convaiinnovations/laya"
# A pre-converted MLX float16 export of the english checkpoint. It loads in ~0.8 s instead of
# converting the bundle in memory on every start, which is worth its ~850 MB.
MLX_REPO = "aac6fef/laya-mlx"

CHECKPOINTS = ("english", "multilingual", "typed-decisions")

# The files each loader opens. `mlx_config.json` is written by `laya_mlx.convert` and read by
# the snake policy module; `manifest.json` ships alongside it.
_LOADER_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*")
_MLX_EXTRA = ("mlx_config.json", "manifest.json")

# The minimum a checkpoint needs before we treat it as vendored.
_REQUIRED = ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json")


def loader_patterns(subfolder: str | None = None) -> list[str]:
    """Hub patterns restricted to loader files, scoped to a single checkpoint."""
    prefix = f"{subfolder}/" if subfolder else ""
    return [prefix + name for name in (*_LOADER_FILES, *_MLX_EXTRA)]


def checkpoint_dir(subfolder: str | None) -> Path:
    """Where a bundled checkpoint lives on disk.

    `english` is the bundle root. Giving each checkpoint its own sibling directory would
    duplicate the shared sentencepiece tokenizer, so the siblings nest under the root --
    which is exactly what the loaders expect from `models/laya` plus a `subfolder`.
    """
    return MODELS / "laya" if subfolder is None else MODELS / "laya" / subfolder


def is_complete(target: Path) -> bool:
    """True when every file the loader requires is present and the weights look whole."""
    if not all((target / name).is_file() for name in _REQUIRED):
        return False
    # A truncated safetensors download is the realistic failure mode, and it is the one that
    # only surfaces as a confusing error at load time, so check the weight file's size.
    return (target / "model.safetensors").stat().st_size > 1_000_000


def checkpoint_size(target: Path) -> int:
    """Bytes of the loader files at this level only, ignoring nested checkpoints."""
    total = 0
    for name in (*_LOADER_FILES, *_MLX_EXTRA):
        if name.endswith("/*"):
            total += sum(p.stat().st_size for p in target.glob(name) if p.is_file())
        elif (target / name).is_file():
            total += (target / name).stat().st_size
    return total


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1000:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.2f} GB"


def ensure(
    repo: str, subfolder: str | None, local_dir: Path, target: Path, label: str, *, force: bool
) -> bool:
    """Download one checkpoint into `target`. Returns True if it did work.

    `local_dir` is the directory the repo is materialised into and `target` is where this
    particular checkpoint ends up inside it; they differ for the bundled siblings, because
    snapshot_download preserves the `subfolder/` prefix of the patterns relative to
    local_dir. Passing the subfolder as local_dir would nest it one level too deep.
    """
    if is_complete(target) and not force:
        print(f"  {label:<18} present     {human(checkpoint_size(target)):>9}")
        return False

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo,
        allow_patterns=loader_patterns(subfolder),
        local_dir=str(local_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    # snapshot_download leaves a .cache/huggingface metadata tree inside local_dir. That is
    # Hub bookkeeping rather than checkpoint data, so keep it out of the vendored tree.
    shutil.rmtree(local_dir / ".cache", ignore_errors=True)
    print(f"  {label:<18} fetched     {human(checkpoint_size(target)):>9}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vendor the Laya checkpoints into ./models for offline use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        default="english,multilingual",
        help="comma-separated subset of %s, or 'all' (default: english,multilingual)"
        % ",".join(CHECKPOINTS),
    )
    parser.add_argument("--no-mlx", action="store_true", help="skip the pre-converted MLX export")
    parser.add_argument("--force", action="store_true", help="re-fetch checkpoints already present")
    args = parser.parse_args(argv)

    wanted = [name.strip() for name in args.checkpoints.split(",") if name.strip()]
    if "all" in wanted:
        wanted = list(CHECKPOINTS)
    unknown = [name for name in wanted if name not in CHECKPOINTS]
    if unknown:
        parser.error(
            "unknown checkpoint(s): %s (known: %s)" % (", ".join(unknown), ", ".join(CHECKPOINTS))
        )

    # (repo, subfolder, local_dir, target, label) -- subfolder scopes the patterns, local_dir
    # is the repo root on disk, target is where this checkpoint lands inside it.
    bundle_root = MODELS / "laya"
    jobs = []
    for name in wanted:
        subfolder = None if name == "english" else name
        jobs.append((BUNDLE_REPO, subfolder, bundle_root, checkpoint_dir(subfolder), name))
    if not args.no_mlx:
        jobs.append((MLX_REPO, None, MODELS / "laya-mlx", MODELS / "laya-mlx", "english (mlx)"))

    print(f"Vendoring into {MODELS.relative_to(ROOT)}/")
    for repo, subfolder, local_dir, target, label in jobs:
        print(f"  source            {repo}")
        ensure(repo, subfolder, local_dir, target, label, force=args.force)

    total = sum(checkpoint_size(target) for *_, target, _ in jobs)
    incomplete = [label for *_, target, label in jobs if not is_complete(target)]
    print(f"Total vendored: {human(total)}")
    if incomplete:
        print("INCOMPLETE: " + ", ".join(incomplete), file=sys.stderr)
        return 1
    print("Next: uv run scripts/verify_offline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

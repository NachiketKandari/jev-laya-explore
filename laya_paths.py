"""Where this project's Laya checkpoints live, and which one a runtime should open.

The weights are vendored under `./models` (see `scripts/fetch_models.py`), so every
script here runs with no network. Both loaders accept a plain local directory and then
append `subfolder` to it, which is how the bundled siblings are reached:

    models/laya                     english          (the bundle root)
    models/laya/multilingual        multilingual
    models/laya/typed-decisions     typed-decisions
    models/laya-mlx                 english, pre-converted to MLX fp16

This module has no imports beyond the standard library, so `demo_router.py` can use it
without dragging PyTorch into an MLX-only process.

Point `LAYA_MODELS_DIR` somewhere else to use a different vendored tree; delete or
rename `models/` and every caller falls back to the Hugging Face Hub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

MODELS_DIR = Path(
    os.environ.get("LAYA_MODELS_DIR") or Path(__file__).resolve().parent / "models"
)
BUNDLE_DIR = MODELS_DIR / "laya"
MLX_EXPORT_DIR = MODELS_DIR / "laya-mlx"

# The Hub ids, used when nothing is vendored.
DEFAULT_MODEL = "convaiinnovations/laya"
DEFAULT_MLX_MODEL = "aac6fef/laya-mlx"

# Files both loaders require before they will accept a directory.
_REQUIRED = ("model.safetensors", "rl_agent_config.json", "encoder/config.json")

# `(model_arg, subfolder)` exactly as the loaders take them.
ModelSpec = Tuple[str, Optional[str]]


def is_checkpoint(path: Path) -> bool:
    """True when `path` holds a complete checkpoint both loaders can open."""
    return bool(path.is_dir()) and all((path / name).is_file() for name in _REQUIRED)


def bundled_checkpoint(subfolder: Optional[str] = None) -> Optional[Path]:
    """Path to a vendored checkpoint, or None when that one isn't vendored."""
    path = BUNDLE_DIR if subfolder is None else BUNDLE_DIR / subfolder
    return path if is_checkpoint(path) else None


def model_for(subfolder: Optional[str] = None, *, mlx: bool = False) -> ModelSpec:
    """Pick the checkpoint to open, preferring vendored weights over the Hub.

    Returns the `(model, subfolder)` pair to hand to `laya.load()` / `laya_mlx.load()`.
    MLX has a pre-converted english export; the other checkpoints are converted from the
    bundled weights in memory at load time.
    """
    if mlx and subfolder is None and is_checkpoint(MLX_EXPORT_DIR):
        return str(MLX_EXPORT_DIR), None
    if bundled_checkpoint(subfolder) is not None:
        return str(BUNDLE_DIR), subfolder
    # Not vendored (or the wrong subfolder): let the loader download from the Hub. MLX ships
    # a standalone pre-converted export of english, so prefer that over converting the
    # original on the fly.
    if mlx and subfolder is None:
        return DEFAULT_MLX_MODEL, None
    return DEFAULT_MODEL, subfolder


def describe() -> str:
    """One-line summary of what is vendored, for `--help` output and verification."""
    if not is_checkpoint(BUNDLE_DIR):
        return f"no vendored weights ({MODELS_DIR} is empty); loaders will use the Hub"
    have = [name for name in ("english", "multilingual", "typed-decisions")
            if bundled_checkpoint(None if name == "english" else name) is not None]
    mlx_note = " + mlx english export" if is_checkpoint(MLX_EXPORT_DIR) else ""
    return f"vendored {MODELS_DIR}: {', '.join(have)}{mlx_note}"

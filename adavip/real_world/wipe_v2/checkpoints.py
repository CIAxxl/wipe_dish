"""Helpers for retaining published Wipe V2 checkpoints."""

import os
from pathlib import Path
import shutil


def retain_checkpoint(source: Path, target: Path) -> None:
    """Publish a copy of a complete checkpoint under a stable filename."""
    pending = target.with_name(f".{target.name}.pending")
    try:
        shutil.copyfile(source, pending)
        os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)

"""Publish atomic AT checkpoints without altering native training or ranking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from reactive_diffusion_policy.workspace.train_at_workspace import TrainATWorkspace

from .checkpoints import retain_checkpoint


class WipeV2ATWorkspace(TrainATWorkspace):
    """Retain final latest and native train-loss top checkpoint reliably."""

    def save_checkpoint(self, path: str | Path | None = None, tag: str = 'latest',
                        exclude_keys: Any = None, include_keys: Any = None,
                        use_thread: bool = True) -> str:
        """Write a complete native payload before replacing the public filename."""
        del use_thread
        target = Path(path) if path is not None else Path(self.get_checkpoint_path(tag))
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name('.' + target.name + '.pending')
        try:
            super().save_checkpoint(path=pending, exclude_keys=exclude_keys,
                                    include_keys=include_keys, use_thread=False)
            os.replace(pending, target)
        finally:
            pending.unlink(missing_ok=True)
        if path is not None and target.name.startswith('epoch='):
            retain_checkpoint(target, target.with_name('best.ckpt'))
        return str(target)

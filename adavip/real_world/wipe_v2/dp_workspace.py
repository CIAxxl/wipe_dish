"""Use the native DP loop with atomic checkpoints and an explicit best alias."""

from __future__ import annotations

import os
import faulthandler
from pathlib import Path
from typing import Any

from reactive_diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace

from .checkpoints import retain_checkpoint


class WipeV2DPWorkspace(TrainDiffusionUnetImageWorkspace):
    """Preserve native optimization and atomically publish complete checkpoint files."""

    def run(self) -> None:
        """Expose stalled native startup stacks without changing the training loop."""
        import numcodecs.blosc

        # Keep compression workers independent of PyTorch/OpenMP thread pools.
        numcodecs.blosc.set_nthreads(1)
        trace = os.environ.get('WIPE_DP_TRACE_STARTUP') == '1'
        if trace:
            faulthandler.dump_traceback_later(120, repeat=True)
        try:
            super().run()
        finally:
            if trace:
                faulthandler.cancel_dump_traceback_later()

    def save_checkpoint(
        self, path: str | Path | None = None, tag: str = 'latest',
        exclude_keys: Any = None, include_keys: Any = None, use_thread: bool = True,
    ) -> str:
        """Retain scheduled epochs and update best when validation loss improves."""
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
        if path is None and tag == 'latest':
            completed_epoch = self.epoch + 1
            if completed_epoch in self.cfg.checkpoint.get('keep_epochs', ()):
                retain_checkpoint(target, target.with_name(f'epoch-{completed_epoch:04d}.ckpt'))
        return str(target.absolute())

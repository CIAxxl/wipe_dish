"""Native LDP optimization with a frozen selected AT."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import dill
import torch
from adavip.real_world.piper_rdp_at import PiperVAE
from adavip.real_world.piper_rdp_dataset import PiperRdpLatentDataset
from adavip.real_world.wipe_v2.dataset import WipeV2Dataset
from adavip.real_world.wipe_v2.dp_workspace import WipeV2DPWorkspace


class WipeV2LatentDataset(WipeV2Dataset, PiperRdpLatentDataset):
    """Fit latent bounds on Wipe V2's publisher training split."""


class WipeFrozenAT(PiperVAE):
    """Load the complete native AT checkpoint and prevent tokenizer weight updates."""

    def __init__(self, **kwargs: Any) -> None:
        checkpoint = kwargs.pop('load_dir')
        assert checkpoint is not None
        super().__init__(load_dir=None, **kwargs)
        payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
        self.load_state_dict(payload)
        for name in ('encoder', 'decoder', 'quant', 'post_quant'):
            getattr(self, name).requires_grad_(False)
        self.eval()

    def train(self) -> None:
        """Keep the fixed tokenizer and decoder in evaluation mode."""
        self.eval()


def assert_frozen(at: PiperVAE, expected: dict) -> None:
    """Require exact original AT weights and input normalization after optimization."""
    for name in ('encoder', 'decoder', 'quant', 'post_quant'):
        module = getattr(at, name)
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[name][key].cpu(), rtol=0, atol=0)
    current = at.normalizer.state_dict()
    for key, value in expected['normalizer'].items():
        torch.testing.assert_close(current[key].cpu(), value.cpu(), rtol=0, atol=0)


class WipeV2LDPWorkspace(WipeV2DPWorkspace):
    """Preserve native training and record the provenance of the external AT."""

    include_keys = ['global_step', 'epoch', 'at_provenance']

    def __init__(self, cfg: Any, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir=output_dir)
        checkpoint = Path(cfg.at_load_dir)
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill, weights_only=False)
        self.at_provenance = dict(path=str(checkpoint), sha256=digest,
                                  completed_epoch=dill.loads(payload['pickles']['epoch']) + 1)
        self._at_expected = payload['state_dicts']['model']
        # Upstream VAE is a plain object: policy.state_dict() does not include it.
        self.frozen_at = self.model.at
        self.ema_frozen_at = self.ema_model.at
        assert_frozen(self.frozen_at, self._at_expected)
        assert_frozen(self.ema_frozen_at, self._at_expected)

    def save_checkpoint(
        self, path: str | Path | None = None, tag: str = 'latest',
        exclude_keys: Any = None, include_keys: Any = None, use_thread: bool = True,
    ) -> str:
        """Check the frozen AT before publishing training checkpoints."""
        assert_frozen(self.frozen_at, self._at_expected)
        assert_frozen(self.ema_frozen_at, self._at_expected)
        return super().save_checkpoint(path=path, tag=tag, exclude_keys=exclude_keys,
                                       include_keys=include_keys, use_thread=use_thread)

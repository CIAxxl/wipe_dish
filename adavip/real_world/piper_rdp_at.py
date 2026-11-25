"""AT adapter that preserves normalization in native RDP checkpoints."""

from __future__ import annotations

import copy
from typing import Any

import torch

from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.model.vae.model import VAE

from adavip.real_world.piper_rdp_normalization import reuse_at_normalizer


class PiperVAE(VAE):
    """Save and restore AT weights together with their input normalization."""

    def __init__(self, shape_meta: dict[str, Any], **kwargs: Any) -> None:
        self.normalizer_checkpoint_loaded = False
        self.normalizer_shapes = {"action": shape_meta["action"]["shape"]}
        self.normalizer_shapes.update({
            key: spec["shape"] for key, spec in shape_meta.get("extended_obs", {}).items()
        })
        super().__init__(shape_meta=shape_meta, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Include normalization omitted by the upstream VAE serializer."""
        state = super().state_dict()
        state["normalizer"] = self.normalizer.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load raw AT states or workspace payloads, requiring input scales."""
        state = state_dict.get("state_dicts", {}).get("model", state_dict)
        saved = LinearNormalizer()
        if not state.get("normalizer"):
            raise ValueError("AT checkpoint has no normalizer; use a newly trained PiperVAE")
        saved.load_state_dict(state["normalizer"])
        for key in ["action"] + list(getattr(self, "extented_obs_keys", [])):
            if key not in saved.params_dict:
                raise ValueError(f"AT checkpoint has no normalizer for {key}")
            field_state = saved[key].state_dict()
            required = {"params_dict." + name for name in (
                "scale", "offset", "input_stats.min", "input_stats.max",
                "input_stats.mean", "input_stats.std",
            )}
            if not required.issubset(field_state):
                raise ValueError(f"AT checkpoint normalizer is incomplete for {key}")
            for name in required:
                value = field_state[name]
                if list(value.shape) != list(self.normalizer_shapes[key]):
                    raise ValueError(f"AT checkpoint normalizer shape mismatch for {key}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"AT checkpoint has non-finite normalization for {key}")
            if torch.any(field_state["params_dict.scale"] <= 0):
                raise ValueError(f"AT checkpoint has invalid normalization scale for {key}")
        super().load_state_dict(state)
        self.normalizer.load_state_dict(saved.state_dict())
        self.normalizer.to(self.device)
        self.normalizer_checkpoint_loaded = True

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Preserve checkpoint affine maps when a workspace supplies scales."""
        if self.normalizer_checkpoint_loaded:
            # Copy before merging: workspace/model normalizers may share callers.
            normalizer = reuse_at_normalizer(
                copy.deepcopy(normalizer), self.normalizer,
                list(getattr(self, "extented_obs_keys", [])),
            )
        super().set_normalizer(normalizer)
        self.normalizer.to(self.device)

    def to(self, device: str | torch.device) -> "PiperVAE":
        """Move normalization along with the AT encoder and decoder."""
        super().to(device)
        self.normalizer.to(device)
        return self

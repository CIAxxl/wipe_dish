"""Dual-arm Wipe action-space DP with explicit raw-time execution alignment."""

from __future__ import annotations

import copy
from typing import Any

import torch

from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class WipeActionDP(DiffusionUnetImagePolicy):
    """Keep the native DP objective and align output to the latest raw observation."""

    def __init__(self, action_start: int = 3, **kwargs: Any) -> None:
        shape_meta = copy.deepcopy(kwargs["shape_meta"])
        if shape_meta.get("extended_obs"):
            raise ValueError("DP cannot receive a future tactile conditioning sequence")
        if list(shape_meta["action"]["shape"]) != [14]:
            raise ValueError("Expected fourteen absolute dual-arm action fields")
        required = {"robot_qpos": 14, "gelsight_left_marker_offset_emb": 15,
                    "gelsight_right_marker_offset_emb": 15}
        expected_keys = set(required) | {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        if set(shape_meta["obs"]) != expected_keys:
            raise ValueError("Wipe DP requires three cameras, dual-arm qpos and two PCA15 inputs")
        for key, width in required.items():
            if (list(shape_meta["obs"][key]["shape"]) != [width]
                    or shape_meta["obs"][key].get("type", "low_dim") != "low_dim"):
                raise ValueError("Invalid Wipe observation shape: " + key)
        for key in expected_keys - set(required):
            spec = shape_meta["obs"][key]
            if spec.get("type") != "rgb" or len(spec["shape"]) != 3 or spec["shape"][0] != 3:
                raise ValueError("Expected a three-channel Wipe camera: " + key)
        kwargs["shape_meta"] = shape_meta
        super().__init__(**kwargs)
        if (self.horizon, self.n_obs_steps, self.n_action_steps, action_start) != (32, 2, 29, 3):
            raise ValueError("Require raw observations1/3 and executable actions3..31")
        if not self.obs_as_global_cond or self.noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("Reviewed DP uses observation conditioning and the epsilon objective")
        self.action_start = action_start
        self.observation_shapes = {key: tuple(spec["shape"]) for key, spec in shape_meta["obs"].items()}

    def validate_observations(self, observations: dict[str, torch.Tensor]) -> None:
        """Reject extra modalities or future frames instead of silently truncating them."""
        if set(observations) != set(self.observation_shapes):
            raise ValueError("Observation keys differ from the reviewed DP configuration")
        for key, shape in self.observation_shapes.items():
            if tuple(observations[key].shape[1:]) != (2, *shape):
                raise ValueError("Expected exactly two available observation frames: " + key)
            if observations[key].shape[0] != observations["robot_qpos"].shape[0]:
                raise ValueError("Observation batch sizes disagree: " + key)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Save train-only raw normalization without trainable scaling parameters."""
        super().set_normalizer(normalizer)
        self.normalizer.requires_grad_(False)

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Use the upstream action-noise MSE without any latent tokenizer."""
        self.validate_observations(batch["obs"])
        if batch.get("extended_obs"):
            raise ValueError("DP must not use recorded future tactile")
        if tuple(batch["action"].shape[1:]) != (32, 14):
            raise ValueError("Expected an unconverted32x14 raw action target")
        return super().compute_loss(batch)

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Execute from raw index3, independently of the two slow observation frames."""
        self.validate_observations(obs_dict)
        prediction = super().predict_action(obs_dict)
        prediction["action"] = prediction["action_pred"][:, self.action_start:self.action_start + self.n_action_steps]
        if tuple(prediction["action"].shape[1:]) != (29, 14):
            raise ValueError("Dual-arm DP execution window mismatch")
        return prediction

"""Offline Piper datasets for RDP action-tokenizer and latent-policy training."""

from __future__ import annotations

import copy
import os
from typing import Any

import einops
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from reactive_diffusion_policy.model.vae.model import VAE

from adavip.real_world.piper_rdp_normalization import (
    fit_train_normalizer,
    load_fixed_split,
    reuse_at_normalizer,
)


class PiperRdpDataset(BaseImageDataset):
    """Sample Piper joint actions and full-rate tactile conditioning."""

    def __init__(
        self,
        shape_meta: dict[str, Any],
        dataset_path: str,
        horizon: int = 29,
        pad_before: int = 3,
        pad_after: int = 25,
        n_obs_steps: int | None = 4,
        obs_temporal_downsample_ratio: int = 2,
        n_latency_steps: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        normalizer_mode: str = "legacy",
        split_path: str | None = None,
        delta_action: bool = False,
        relative_action: bool = False,
    ) -> None:
        if normalizer_mode not in {"legacy", "train_only_joint"}:
            raise ValueError(f"Unknown normalizer_mode: {normalizer_mode}")
        if delta_action or relative_action:
            raise ValueError("PiperRdpDataset supports absolute joint actions only")
        if normalizer_mode == "train_only_joint":
            if max_train_episodes is not None:
                raise ValueError("Fixed split requires max_train_episodes=null")
            if list(shape_meta["action"]["shape"]) != [14]:
                raise ValueError("train_only_joint requires 14D absolute dual-arm actions")
        self.normalizer_mode = normalizer_mode
        self._is_validation = False
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
        if obs_temporal_downsample_ratio < 1:
            raise ValueError("obs_temporal_downsample_ratio must be positive")
        if n_obs_steps is not None and n_obs_steps % obs_temporal_downsample_ratio:
            raise ValueError(
                "n_obs_steps must be divisible by obs_temporal_downsample_ratio"
            )

        self.shape_meta = shape_meta
        self.rgb_keys, self.lowdim_keys = self._split_keys(shape_meta["obs"])
        self.extended_rgb_keys, self.extended_lowdim_keys = self._split_keys(
            shape_meta.get("extended_obs", {})
        )
        load_keys = list(
            dict.fromkeys(
                self.rgb_keys
                + self.lowdim_keys
                + self.extended_rgb_keys
                + self.extended_lowdim_keys
                + ["action"]
            )
        )
        if normalizer_mode == "train_only_joint":
            if self.extended_rgb_keys:
                raise ValueError("AT temporal conditions must be low-dimensional features")
            # Validate metadata before copying potentially large RGB arrays.
            source = ReplayBuffer.create_from_path(
                os.path.join(dataset_path, "replay_buffer.zarr"), mode="r"
            )
            saved_train, saved_val = load_fixed_split(
                split_path or os.path.join(dataset_path, "split.json"),
                source.meta, seed, val_ratio,
            )
            if not np.array_equal(
                get_val_mask(source.n_episodes, val_ratio, seed), saved_val
            ):
                raise ValueError("Generated episode partition disagrees with split.json")
        replay_buffer = ReplayBuffer.copy_from_path(
            os.path.join(dataset_path, "replay_buffer.zarr"), keys=load_keys
        )
        key_first_k: dict[str, int] = {}
        if n_obs_steps is not None:
            extended_keys = set(self.extended_rgb_keys + self.extended_lowdim_keys)
            for key in self.rgb_keys + self.lowdim_keys:
                if key not in extended_keys:
                    key_first_k[key] = n_obs_steps

        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.n_latency_steps = n_latency_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed)
        if normalizer_mode == "train_only_joint":
            train_mask, val_mask = saved_train, saved_val
        self.train_mask = train_mask.copy()
        self.sampler = self._make_sampler(train_mask)
        self.val_mask = val_mask

    @staticmethod
    def _split_keys(shape_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        rgb_keys: list[str] = []
        lowdim_keys: list[str] = []
        for key, attributes in shape_meta.items():
            observation_type = attributes.get("type", "low_dim")
            if observation_type == "rgb":
                rgb_keys.append(key)
            elif observation_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(
                    f"Unsupported observation type {observation_type!r} for {key}"
                )
        return rgb_keys, lowdim_keys

    def _make_sampler(self, episode_mask: np.ndarray) -> SequenceSampler:
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            key_first_k=self.key_first_k,
        )

    def get_validation_dataset(self) -> "PiperRdpDataset":
        """Return a shallow copy sampling validation episodes only."""
        validation = copy.copy(self)
        validation.sampler = validation._make_sampler(self.val_mask)
        validation.val_mask = ~self.val_mask
        validation._is_validation = True
        return validation

    def get_normalizer(self, **kwargs: Any) -> LinearNormalizer:
        """Fit joint/tactile scales, using only fixed training rows in new mode."""
        del kwargs
        if self.normalizer_mode == "train_only_joint":
            return fit_train_normalizer(
                self.replay_buffer, self.replay_buffer.episode_ends,
                self.train_mask, self.shape_meta,
            )
        normalizer = LinearNormalizer()
        action_dim = int(self.shape_meta["action"]["shape"][0])
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"][:, :action_dim]
        )
        for key in sorted(set(self.lowdim_keys + self.extended_lowdim_keys)):
            if key in self.shape_meta["obs"]:
                metadata = self.shape_meta["obs"][key]
            else:
                metadata = self.shape_meta["extended_obs"][key]
            dimension = int(metadata["shape"][0])
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key][:, :dimension]
            )
        for key in sorted(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return the configured absolute joint and gripper actions."""
        action_dim = int(self.shape_meta["action"]["shape"][0])
        return torch.from_numpy(self.replay_buffer["action"][:, :action_dim])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> dict[str, Any]:
        threadpool_limits(1)
        # Upstream's first-k optimization fills unloaded uint8 image slots with
        # NaN before slicing them away, which emits a harmless NumPy cast warning.
        with np.errstate(invalid="ignore"):
            data = self.sampler.sample_sequence(index)
        observation_slice = slice(self.n_obs_steps)
        ratio = self.obs_temporal_downsample_ratio

        obs: dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            values = data[key][observation_slice][::-ratio][::-1]
            obs[key] = torch.from_numpy(
                np.moveaxis(values, -1, 1).astype(np.float32) / 255.0
            )
        for key in self.lowdim_keys:
            dimension = int(self.shape_meta["obs"][key]["shape"][0])
            values = data[key][:, :dimension][observation_slice][::-ratio][::-1]
            obs[key] = torch.from_numpy(values.astype(np.float32))

        extended_obs: dict[str, torch.Tensor] = {}
        for key in self.extended_rgb_keys:
            extended_obs[key] = torch.from_numpy(
                np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            )
        for key in self.extended_lowdim_keys:
            dimension = int(self.shape_meta["extended_obs"][key]["shape"][0])
            extended_obs[key] = torch.from_numpy(
                data[key][:, :dimension].astype(np.float32)
            )

        action_dim = int(self.shape_meta["action"]["shape"][0])
        action = data["action"][:, :action_dim].astype(np.float32)
        if self.n_latency_steps:
            action = action[self.n_latency_steps :]
        return {
            "obs": obs,
            "action": torch.from_numpy(action),
            "extended_obs": extended_obs,
        }


class PiperRdpLatentDataset(PiperRdpDataset):
    """Piper dataset that also fits an AT latent-action normalizer for LDP."""

    def __init__(
        self,
        at: VAE,
        use_latent_action_before_vq: bool = False,
        latent_normalizer_batch_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if latent_normalizer_batch_size < 1:
            raise ValueError("latent_normalizer_batch_size must be positive")
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq
        self.latent_normalizer_batch_size = latent_normalizer_batch_size

    def get_normalizer(self, **kwargs: Any) -> LinearNormalizer:
        """Fit observation/action normalizers and encoded latent-action bounds."""
        if self.normalizer_mode == "train_only_joint":
            if self._is_validation:
                raise ValueError("Latent normalizer must be fitted on the training dataset")
            if not getattr(self.at, "normalizer_checkpoint_loaded", False):
                raise ValueError(
                    "train_only_joint LDP requires a loaded PiperVAE AT checkpoint "
                    "with its saved normalizer"
                )
            extended_keys = self.extended_lowdim_keys + self.extended_rgb_keys
            if set(getattr(self.at, "extented_obs_keys", [])) != set(extended_keys):
                raise ValueError("AT decoder condition keys disagree with the LDP dataset")
        normalizer = super().get_normalizer(**kwargs)
        if self.normalizer_mode == "train_only_joint":
            normalizer = reuse_at_normalizer(
                normalizer, self.at.normalizer, extended_keys,
            )
        self.at.eval()
        action_dim = int(self.shape_meta["action"]["shape"][0])
        action_sampler = copy.copy(self.sampler)
        action_sampler.keys = ["action"]
        latent_actions: list[np.ndarray] = []
        with torch.no_grad():
            for start in tqdm(
                range(0, len(action_sampler), self.latent_normalizer_batch_size),
                leave=False,
                desc="Calculating latent action normalizer",
            ):
                stop = min(
                    start + self.latent_normalizer_batch_size, len(action_sampler)
                )
                action_batch = []
                for index in range(start, stop):
                    action = action_sampler.sample_sequence(index)["action"]
                    action = action[:, :action_dim].astype(np.float32)
                    if self.n_latency_steps:
                        action = action[self.n_latency_steps :]
                    action_batch.append(action)
                action = torch.from_numpy(np.stack(action_batch))
                # The train-only normalizer is deliberately kept on CPU. Its
                # normalize() moves inputs to its own device, so transfer AFTER
                # normalization before invoking a GPU-resident AT encoder.
                normalized_action = normalizer["action"].normalize(action).to(self.at.device)
                latent = self.at.encoder(
                    self.at.preprocess(normalized_action / self.at.act_scale)
                )
                if self.at.use_vq:
                    if not self.use_latent_action_before_vq:
                        latent, _, _ = self.at.quant_state_with_vq(latent)
                    elif self.at.use_conv_encoder:
                        latent = einops.rearrange(latent, "N T A -> N (T A)")
                else:
                    latent, _ = self.at.quant_state_without_vq(latent)
                latent = einops.rearrange(
                    latent,
                    "N (T A) -> N T A",
                    T=self.at.downsampled_input_h,
                )
                latent_array = latent.detach().cpu().numpy()
                latent_actions.append(latent_array.reshape(-1, latent_array.shape[-1]))
        all_latent_actions = np.concatenate(latent_actions, axis=0)
        if not np.isfinite(all_latent_actions).all():
            raise ValueError("Non-finite training latent actions")
        normalizer["latent_action"] = SingleFieldLinearNormalizer.create_fit(
            all_latent_actions
        )
        return normalizer

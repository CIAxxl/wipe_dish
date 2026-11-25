"""Piper compatibility policy for upstream latent RDP."""

from __future__ import annotations

from typing import Dict

import einops
import torch

from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import (
    LatentDiffusionUnetImagePolicy,
)


class PiperLatentDiffusionUnetImagePolicy(LatentDiffusionUnetImagePolicy):
    """Decode latent actions using the original Piper action dimension.

    Upstream overwrites ``action_dim`` with the latent width during policy
    construction. That width is correct for diffusion but not for the decoded
    action tensor when a dual-arm action has more dimensions than the latent.
    """

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        dataset_obs_temporal_downsample_ratio: int,
        extended_obs_dict: Dict[str, torch.Tensor] | None = None,
        return_latent_action: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample a latent trajectory and decode all original action fields."""
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is unavailable")
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        latent_horizon = self.horizon
        latent_dim = self.action_dim
        obs_steps = self.n_obs_steps

        this_nobs = dict_apply(
            nobs,
            lambda value: value[:, :obs_steps, ...].reshape(-1, *value.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)
        global_cond = obs_features.reshape(batch_size, -1)
        cond_data = torch.zeros(
            (batch_size, latent_horizon, latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        latent_sample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )
        latent_sample = self.normalizer["latent_action"].unnormalize(latent_sample)
        latent_sample = einops.rearrange(latent_sample, "N T A -> N (T A)")
        if self.at.use_vq:
            if self.use_latent_action_before_vq:
                decoded_latent, _, _ = self.at.quant_state_with_vq(latent_sample)
            else:
                decoded_latent = latent_sample
        else:
            decoded_latent = self.at.postprocess_quant_state_without_vq(latent_sample)

        if return_latent_action:
            action_pred = decoded_latent.unsqueeze(1).expand(
                -1, self.original_horizon, -1
            )
        else:
            if self.at.use_rnn_decoder:
                if extended_obs_dict is None:
                    raise ValueError("extended_obs_dict is required by the AT decoder")
                temporal_cond = self.at.get_temporal_cond(extended_obs_dict)
                decoded_action = self.at.get_action_from_latent_with_temporal_cond(
                    decoded_latent, temporal_cond.to(self.device)
                )
            else:
                decoded_action = self.at.get_action_from_latent(decoded_latent)
            original_action_dim = self.at.input_dim_w
            normalized_action = decoded_action[..., :original_action_dim]
            action_pred = self.normalizer["action"].unnormalize(normalized_action)

        aligned_obs_steps = obs_steps * dataset_obs_temporal_downsample_ratio
        start = aligned_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def predict_from_latent_action(
        self,
        latent_action: torch.Tensor,
        extended_obs_dict: Dict[str, torch.Tensor],
        extended_obs_last_step: int,
        dataset_obs_temporal_downsample_ratio: int,
        extend_obs_pad_after: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode a token incrementally using the original action dimension."""
        latent_action = latent_action.to(self.device)
        if self.at.use_rnn_decoder:
            pad_after = (
                self.original_horizon - extended_obs_last_step
                if extend_obs_pad_after
                else None
            )
            temporal_cond = self.at.get_temporal_cond(
                extended_obs_dict,
                extended_obs_last_step,
                extend_obs_pad_after_n=pad_after,
            )
            decoded_action = self.at.get_action_from_latent_with_temporal_cond(
                latent_action, temporal_cond.to(self.device)
            )
        else:
            decoded_action = self.at.get_action_from_latent(latent_action)

        original_action_dim = self.at.input_dim_w
        normalized_action = decoded_action[..., :original_action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        aligned_obs_steps = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        start = aligned_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

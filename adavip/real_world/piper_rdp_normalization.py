"""Fixed-split normalization for dual-arm absolute joint RDP datasets."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def load_fixed_split(
    path: str | Path,
    meta: Any,
    seed: int,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate saved episode IDs, order, boundaries, and complete partition."""
    with Path(path).open(encoding="utf-8") as stream:
        split = json.load(stream)
    ends = np.asarray(meta["episode_ends"])
    if "episode_ids" not in meta:
        raise ValueError("Fixed split requires meta/episode_ids")
    ids = np.asarray(meta["episode_ids"])
    if (
        ends.ndim != 1 or not len(ends)
        or not np.issubdtype(ends.dtype, np.integer)
        or np.any(np.diff(np.r_[0, ends]) <= 0)
        or ids.shape != ends.shape
        or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Invalid episode boundaries or duplicate/invalid episode IDs")
    if split.get("seed") != seed or split.get("val_ratio") != val_ratio:
        raise ValueError("Configured seed/val_ratio disagree with split.json")
    masks = []
    for name, saved_mask_key in (("train", "train_mask"), ("validation", "val_mask")):
        indices = np.asarray(split.get(name + "_episode_indices", []))
        expected_ids = np.asarray(split.get(name + "_episode_ids", []))
        if (
            indices.ndim != 1 or not indices.size
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(ends))
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(f"Invalid {name} episode indices in split.json")
        if (
            expected_ids.shape != indices.shape
            or not np.issubdtype(expected_ids.dtype, np.integer)
            or not np.array_equal(ids[indices], expected_ids)
        ):
            raise ValueError(f"{name} episode IDs/order disagree with split.json")
        mask = np.zeros(len(ends), dtype=bool)
        mask[indices] = True
        if saved_mask_key in meta and not np.array_equal(meta[saved_mask_key], mask):
            raise ValueError(f"meta/{saved_mask_key} disagrees with split.json")
        masks.append(mask)
    train, validation = masks
    if np.any(train & validation) or not np.all(train | validation):
        raise ValueError("split.json must partition all episodes without overlap")
    return train, validation


def fit_train_normalizer(
    data: Any,
    episode_ends: np.ndarray,
    train_mask: np.ndarray,
    shape_meta: dict[str, Any],
) -> LinearNormalizer:
    """Fit per-channel limits from original training rows, excluding windows."""
    if list(shape_meta["action"]["shape"]) != [14]:
        raise ValueError("train_only_joint requires 14D absolute dual-arm actions")
    ends = np.asarray(episode_ends)
    train_mask = np.asarray(train_mask)
    if (
        ends.ndim != 1 or not ends.size
        or not np.issubdtype(ends.dtype, np.integer)
        or np.any(np.diff(np.r_[0, ends]) <= 0)
        or train_mask.shape != ends.shape or train_mask.dtype != np.bool_
        or not train_mask.any()
    ):
        raise ValueError("Invalid training episode mask or boundaries")
    starts = np.r_[0, ends[:-1]]
    fields = {"action": shape_meta["action"]}
    rgb_keys = set()
    for section in ("obs", "extended_obs"):
        for key, spec in shape_meta.get(section, {}).items():
            if spec.get("type", "low_dim") == "rgb":
                rgb_keys.add(key)
            else:
                if key in fields and fields[key]["shape"] != spec["shape"]:
                    raise ValueError(f"Inconsistent shapes for {key}")
                fields[key] = spec
    if rgb_keys & fields.keys():
        raise ValueError("A field cannot be both RGB and low dimensional")
    normalizer = LinearNormalizer()
    for key, spec in fields.items():
        shape = tuple(spec["shape"])
        if len(shape) != 1 or data[key].shape != (int(ends[-1]),) + shape:
            raise ValueError(f"Replay-buffer shape disagrees with shape_meta for {key}")
        values = np.concatenate([
            np.asarray(data[key][int(start):int(end)], dtype=np.float32)
            for start, end in zip(starts[train_mask], ends[train_mask])
        ])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training values in {key}")
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(values, mode="limits")
    for key in rgb_keys:
        normalizer[key] = get_image_range_normalizer()
    normalizer.requires_grad_(False)
    return normalizer


def reuse_at_normalizer(
    normalizer: LinearNormalizer,
    saved: LinearNormalizer,
    extended_keys: list[str],
) -> LinearNormalizer:
    """Check AT input affine maps and copy the exact checkpoint parameters."""
    for key in dict.fromkeys(["action"] + extended_keys):
        if key not in saved.params_dict or key not in normalizer.params_dict:
            raise ValueError(f"AT checkpoint/current dataset has no normalizer for {key}")
        current_state = normalizer[key].state_dict()
        saved_state = saved[key].state_dict()
        if current_state.keys() != saved_state.keys():
            raise ValueError(f"AT {key} normalizer state structure mismatch")
        for name in current_state:
            current = current_state[name].detach().cpu()
            expected = saved_state[name].detach().cpu()
            if (
                current.shape != expected.shape or not torch.isfinite(expected).all()
                or (name == "params_dict.scale" and torch.any(expected == 0))
                or not torch.allclose(current, expected, rtol=1e-6, atol=1e-7)
            ):
                raise ValueError(f"AT {key} normalization mismatch; check dataset/split")
        normalizer[key] = copy.deepcopy(saved[key]).cpu()
    normalizer.requires_grad_(False)
    return normalizer

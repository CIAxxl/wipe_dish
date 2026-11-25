"""Paired deterministic stratified K64 labels with signed utility and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .uvip_data import UvipDataset, atomic_json, file_hash, fingerprint


STRATA = ((0, 12, 16, .12), (12, 25, 12, .13), (25, 50, 12, .25),
          (50, 75, 12, .25), (75, 100, 12, .25))


def paired_draws(config: dict, window_ids: np.ndarray, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive each window's draws independently of batch size or shard recovery."""
    times, noise = [], []
    for episode, frame in window_ids:
        identity = dict(seed=config["label_seed"], episode=int(episode), frame=int(frame), split=split)
        seed = int(fingerprint(identity)[:15], 16)
        generator = torch.Generator().manual_seed(seed)
        times.append(torch.cat([torch.randint(low, high, (count,), generator=generator)
                                for low, high, count, _ in STRATA]))
        noise.append(torch.randn((64, config["horizon"], config["action_dim"]), generator=generator))
    return torch.stack(times), torch.stack(noise)


def aggregate_draws(differences: torch.Tensor) -> dict[str, torch.Tensor]:
    """Estimate the original timestep expectation, variance and two K32 halves."""
    # differences: [windows, conditions-minus-full, 64]
    estimate = differences.new_zeros(differences.shape[:2])
    variance, half_a, half_b = estimate.clone(), estimate.clone(), estimate.clone()
    start = 0
    for _, _, count, weight in STRATA:
        values = differences[..., start:start + count]
        estimate += weight * values.mean(-1)
        variance += weight ** 2 * values.var(-1, unbiased=True) / count
        half_a += weight * values[..., :count // 2].mean(-1)
        half_b += weight * values[..., count // 2:].mean(-1)
        start += count
    return dict(utility=estimate, variance=variance, half_a=half_a, half_b=half_b)


@torch.no_grad()
def score_windows(policy, batch: dict, windows: np.ndarray, split: str, microbatch: int) -> dict[str, np.ndarray]:
    """Reuse full/removed actions, noise/time and mask; zero whole token blocks."""
    from .uvip_training import move_batch
    device = next(policy.parameters()).device
    times, noise = paired_draws(policy.config, windows, split)
    batch = move_batch(batch, device)
    tokens = policy.encoder.tokens(batch["obs"], policy.normalizer)
    conditionings = [policy.encoder.context(tokens, batch["obs"]["language"],
                                            mask[None].expand(len(tokens), -1))
                    for mask in policy.conditions]
    if policy.config.get("label_vectorized", False):
        losses = _score_paired_batches(policy, batch, conditionings, times, noise, microbatch)
        result = aggregate_draws(losses[:, 1:] - losses[:, :1])
        result["full_loss"] = aggregate_draws(losses[:, :1])["utility"]
        return {key: value.numpy() for key, value in result.items()}
    losses = []
    for condition in conditionings:
        window_losses = []
        for window in range(len(tokens)):
            values = []
            for start in range(0, 64, microbatch):
                stop = min(64, start + microbatch)
                repeated = {key: batch[key][window:window + 1].expand(stop - start, -1, -1)
                            for key in ("action", "action_mask")}
                values.append(policy.denoising_loss(repeated, condition[window:window + 1].expand(stop - start, -1),
                                                    noise[window, start:stop].to(device),
                                                    times[window, start:stop].to(device)).double().cpu())
            window_losses.append(torch.cat(values))
        losses.append(torch.stack(window_losses))
    losses = torch.stack(losses, 1)
    result = aggregate_draws(losses[:, 1:] - losses[:, :1])
    result["full_loss"] = aggregate_draws(losses[:, :1])["utility"]
    return {key: value.numpy() for key, value in result.items()}


def _score_paired_batches(policy, batch: dict, conditionings: list[torch.Tensor],
                         times: torch.Tensor, noise: torch.Tensor, microbatch: int) -> torch.Tensor:
    """Batch real window/condition/draw combinations without changing paired draws."""
    if microbatch < 1:
        raise ValueError("Label microbatch must be positive")
    device = next(policy.parameters()).device
    conditions = torch.stack(conditionings, dim=1)
    window_count, condition_count = conditions.shape[:2]
    # Each window's noise/time bank is reused across full and removed conditions.
    # Keep only one U-Net microbatch on device; the full loss table stays on CPU.
    noise, times = noise.to(device), times.to(device)
    flat_losses = torch.empty(window_count * condition_count * 64, dtype=torch.float64)
    for start in range(0, len(flat_losses), microbatch):
        stop = min(start + microbatch, len(flat_losses))
        indices = torch.arange(start, stop, device=device)
        draw_ids = indices.remainder(64)
        condition_ids = indices.div(64, rounding_mode="floor").remainder(condition_count)
        window_ids = indices.div(64 * condition_count, rounding_mode="floor")
        repeated = {key: batch[key][window_ids] for key in ("action", "action_mask")}
        values = policy.denoising_loss(repeated, conditions[window_ids, condition_ids],
                                       noise[window_ids, draw_ids], times[window_ids, draw_ids])
        flat_losses[start:stop] = values.double().cpu()
    return flat_losses.reshape(window_count, condition_count, 64)


def generate_labels(config: dict, policy, teacher_path: Path, output: Path, split: str,
                    max_shards: int = 0) -> dict:
    """Publish resumable checked shards, then a complete bank with train-only RMS."""
    from .uvip_training import source_fingerprint
    dataset = UvipDataset(config, split)
    identity = dict(teacher_sha256=file_hash(teacher_path), split=split,
                    config=fingerprint(config), source=source_fingerprint(config),
                    index=fingerprint(dataset.index.tolist()), strata=STRATA,
                    condition_names=config["utility_names"], label_seed=config["label_seed"])
    digest = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text())["identity"] != json.loads(json.dumps(identity)):
            raise ValueError("Label continuation contract changed")
    atomic_json(output / "manifest.json", dict(identity=identity, fingerprint=digest))
    policy.eval().requires_grad_(False)
    shard_size = config["label_shard_size"]
    records = []
    for start in range(0, len(dataset), shard_size):
        stop = min(len(dataset), start + shard_size)
        shard = output / f"shard_{start:08d}_{stop:08d}.npz"
        marker = shard.with_suffix(".json")
        if shard.exists() and marker.exists():
            record = json.loads(marker.read_text())
            if record["contract"] != digest or file_hash(shard) != record["sha256"]:
                raise ValueError("Shard checksum/identity mismatch")
        else:
            chunks = []
            for row in range(start, stop, config["label_window_batch"]):
                end = min(stop, row + config["label_window_batch"])
                batch = torch.utils.data.default_collate([dataset[i] for i in range(row, end)])
                chunks.append(score_windows(policy, batch, dataset.index[row:end], split, config["label_microbatch"]))
            arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError("Nonfinite utility labels")
            temporary = shard.with_name(shard.name + ".partial")
            with temporary.open("wb") as stream:
                np.savez(stream, index=dataset.index[start:stop], **arrays)
            temporary.replace(shard)
            record = dict(contract=digest, start=start, stop=stop, sha256=file_hash(shard))
            atomic_json(marker, record)
        records.append(record)
        print(json.dumps(dict(event="label_shard", split=split, stop=stop, windows=len(dataset))), flush=True)
        if max_shards and len(records) >= max_shards:
            return dict(complete=False, shards=len(records))
    arrays = {key: [] for key in ("utility", "variance", "half_a", "half_b", "full_loss")}
    for record in records:
        shard = output / f"shard_{record['start']:08d}_{record['stop']:08d}.npz"
        with np.load(shard, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["index"], dataset.index[record["start"]:record["stop"]])
            for key in arrays:
                arrays[key].append(saved[key])
    joined = {key: np.concatenate(value) for key, value in arrays.items()}
    joined["index"] = dataset.index
    if split == "train":
        joined["rms"] = np.maximum(np.sqrt(np.square(joined["utility"]).mean(0)), 1e-6).astype(np.float32)
    files = {}
    for key, array in joined.items():
        temporary = output / (key + ".npy.partial")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(output / (key + ".npy"))
        files[key + ".npy"] = file_hash(output / (key + ".npy"))
    summary = dict(complete=True, contract=digest, windows=len(dataset), files=files,
                   negative_fraction=(joined["utility"] < 0).mean(0).tolist(),
                   mean_estimator_variance=joined["variance"].mean(0).tolist(),
                   half_difference_rms=np.sqrt(np.square(joined["half_a"] - joined["half_b"]).mean(0)).tolist())
    atomic_json(output / "COMPLETE.json", summary)
    return summary


def validate_bank(path: Path, expected_teacher_hash: str) -> dict:
    """Reject incomplete banks, tampered aggregates or a different Teacher."""
    complete = json.loads((path / "COMPLETE.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    if not complete["complete"] or complete["contract"] != manifest["fingerprint"]:
        raise ValueError("Label bank incomplete")
    if manifest["identity"]["teacher_sha256"] != expected_teacher_hash:
        raise ValueError("Label Teacher differs")
    for name, expected in complete["files"].items():
        if file_hash(path / name) != expected:
            raise ValueError("Label bank aggregate checksum mismatch")
    return complete

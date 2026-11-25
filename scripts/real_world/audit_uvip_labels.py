"""Audit completed UViP labels before authorizing Student GPU checks."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from adavip.real_world.uvip_data import atomic_json, file_hash, fingerprint, make_index
from adavip.real_world.uvip_labels import STRATA, validate_bank
from adavip.real_world.uvip_training import selected_teacher, source_fingerprint


def audit(config: dict) -> dict:
    """Verify exact joins, shard integrity and statistics without training a model."""
    root = Path(config["output_root"])
    teacher = selected_teacher(config)
    teacher_hash = file_hash(teacher)
    completion = json.loads((root / "teacher/COMPLETE.json").read_text())
    if completion["checkpoint_sha256"] != teacher_hash or completion["epochs"] != 100:
        raise ValueError("Teacher completion/selection mismatch")
    report = dict(task=config["task"], passed=False, student_training_submitted=False,
                  verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  teacher_sha256=teacher_hash, config=fingerprint(config),
                  source=source_fingerprint(config), splits={})
    split_episodes = []
    width = len(config["utility_names"])
    for split in ("train", "val"):
        path = root / "labels" / split
        complete = validate_bank(path, teacher_hash)
        manifest = json.loads((path / "manifest.json").read_text())
        identity = manifest["identity"]
        expected = make_index(config, split)
        required = dict(teacher_sha256=teacher_hash, config=report["config"], source=report["source"],
                        split=split, index=fingerprint(expected.tolist()),
                        condition_names=config["utility_names"], label_seed=config["label_seed"],
                        strata=json.loads(json.dumps(STRATA)))
        if identity != required or fingerprint(identity) != manifest["fingerprint"]:
            raise ValueError("Label source/config/index/conditions contract differs")
        expected_files = {name + ".npy" for name in ("utility", "variance", "half_a", "half_b", "full_loss", "index")}
        if split == "train":
            expected_files.add("rms.npy")
        if set(complete["files"]) != expected_files:
            raise ValueError("Label aggregate file set differs")
        arrays = {name: np.load(path / (name + ".npy"), allow_pickle=False)
                  for name in ("utility", "variance", "half_a", "half_b", "full_loss", "index")}
        np.testing.assert_array_equal(arrays["index"], expected)
        if complete["windows"] != len(expected):
            raise ValueError("Label coverage differs")
        split_episodes.append(set(expected[:, 0].tolist()))
        for name, array in arrays.items():
            shape = expected.shape if name == "index" else (len(expected), 1 if name == "full_loss" else width)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {split}/{name} shape or finite values")
        if (arrays["variance"] < 0).any() or (arrays["full_loss"] < 0).any():
            raise ValueError("Negative variance/full loss")
        records = []
        for start in range(0, len(expected), config["label_shard_size"]):
            stop = min(len(expected), start + config["label_shard_size"])
            shard = path / f"shard_{start:08d}_{stop:08d}.npz"
            record = json.loads(shard.with_suffix(".json").read_text())
            if (record["start"], record["stop"], record["contract"]) != (start, stop, manifest["fingerprint"]):
                raise ValueError("Shard cursor/contract differs")
            if file_hash(shard) != record["sha256"]:
                raise ValueError("Shard checksum differs")
            with np.load(shard, allow_pickle=False) as values:
                if set(values.files) != set(arrays):
                    raise ValueError("Shard array set differs")
                for name in arrays:
                    np.testing.assert_array_equal(values[name], arrays[name][start:stop])
            records.append(record)
        if len(list(path.glob("shard_*.npz"))) != len(records):
            raise ValueError("Unexpected extra shards")
        # The two K32 estimates must average exactly to the K64 signed utility.
        np.testing.assert_allclose((arrays["half_a"] + arrays["half_b"]) / 2,
                                   arrays["utility"], rtol=1e-10, atol=1e-12)
        stats = {}
        for column, name in enumerate(config["utility_names"]):
            utility = arrays["utility"][:, column]
            half_difference = arrays["half_a"][:, column] - arrays["half_b"][:, column]
            stats[name] = dict(mean=float(utility.mean()), std=float(utility.std()),
                               quantiles=np.quantile(utility, [0, .05, .5, .95, 1]).tolist(),
                               negative_fraction=float((utility < 0).mean()),
                               utility_rms=float(np.sqrt(np.square(utility).mean())),
                               mean_estimator_variance=float(arrays["variance"][:, column].mean()),
                               half_difference_rms=float(np.sqrt(np.square(half_difference).mean())))
        if split == "train":
            rms = np.load(path / "rms.npy", allow_pickle=False)
            expected_rms = np.maximum(np.sqrt(np.square(arrays["utility"]).mean(0)), 1e-6).astype(np.float32)
            np.testing.assert_array_equal(rms, expected_rms)
            if rms.shape != (width,) or not np.isfinite(rms).all() or (rms <= 0).any():
                raise ValueError("Invalid training-only RMS")
        elif (path / "rms.npy").exists():
            raise ValueError("Validation must not fit its own RMS")
        report["splits"][split] = dict(windows=len(expected), episodes=len(split_episodes[-1]),
                                       shards_verified=len(records), conditions=stats,
                                       complete_sha256=file_hash(path / "COMPLETE.json"))
        print(json.dumps(dict(event="label_audit_split_passed", task=config["task"], split=split,
                              windows=len(expected), shards=len(records))), flush=True)
    if split_episodes[0] & split_episodes[1]:
        raise ValueError("Train/validation episode overlap")
    report["passed"] = True
    report["quality_note"] = "Integrity passed; signed values and sampling diagnostics require review before Student."
    atomic_json(root / "labels/AUDIT.json", report)
    return report


def main() -> None:
    """Check both split banks and write a durable report only after all checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(json.dumps(audit(config)), flush=True)


if __name__ == "__main__":
    main()

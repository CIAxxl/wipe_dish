#!/usr/bin/env python3
"""Prepare, preflight and train independent real-robot UViP task bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from adavip.real_world.uvip_data import UvipDataset, atomic_json, build_image_cache, file_hash, fingerprint, fit_normalizer
from adavip.real_world.uvip_labels import generate_labels
from adavip.real_world.uvip_training import atomic_checkpoint, load_policy, selected_teacher, source_fingerprint, train


def prepare(config: dict) -> None:
    """Check immutable inputs and persist train-only statistics and CPU evidence."""
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    statistics = fit_normalizer(config)
    reports = {}
    for split in ("train", "val"):
        dataset = UvipDataset(config, split, require_cache="wipe_v2_hdf5" in config)
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[index]
            if sample["action"].shape != (config["horizon"], config["action_dim"]):
                raise ValueError("Action shape mismatch")
            if not torch.isfinite(sample["action"]).all() or not torch.isfinite(sample["obs"]["state"]).all():
                raise ValueError("Nonfinite source sample")
        reports[split] = dict(windows=len(dataset), episodes=len(set(dataset.index[:, 0].tolist())))
    atomic_json(root / "normalizer.json", statistics)
    atomic_json(root / "prepared.json", dict(config=fingerprint(config), source=source_fingerprint(config),
                                            splits=reports, gpu_preflight=False))
    print(json.dumps(reports), flush=True)


def preflight(config: dict) -> None:
    """Run the bounded, measured GPU checks without starting formal training."""
    from adavip.real_world.uvip_preflight import run_preflight
    run_preflight(config)


def main() -> None:
    """Dispatch explicit stages; never chain GPU stages implicitly."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "cache", "preflight", "teacher", "labels", "student", "export"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "cache":
        print(build_image_cache(config, Path(config["cache_root"])))
    elif args.stage == "preflight":
        preflight(config)
    elif args.stage in ("teacher", "student"):
        train(config, student=args.stage == "student", resume=args.resume)
    elif args.stage == "labels":
        if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Label generation requires a Slurm GPU step")
        path = selected_teacher(config)
        teacher, _ = load_policy(path, config, "cuda")
        for split in ("train", "val"):
            generate_labels(config, teacher, path, Path(config["output_root"]) / "labels" / split, split)
    else:
        root = Path(config["output_root"])
        complete = json.loads((root / "student/COMPLETE.json").read_text())
        path = root / "student/best.pt"
        if complete["checkpoint_sha256"] != file_hash(path):
            raise ValueError("Student checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        atomic_checkpoint(root / "deployment/student_ema.pt", {key: saved[key] for key in ("config", "student", "ema", "identity", "initialization")})
        atomic_json(root / "deployment/manifest.json", dict(checkpoint_sha256=file_hash(root / "deployment/student_ema.pt"),
                                                            language_sha256=file_hash(config["language_embedding"]),
                                                            action_semantics=config["action_semantics"]))


if __name__ == "__main__":
    main()

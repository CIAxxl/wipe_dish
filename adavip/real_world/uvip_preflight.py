"""Bounded GPU preflight on real task batches, with measured training resources."""

from __future__ import annotations

import copy
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .uvip_data import UvipDataset, atomic_json, file_hash, fingerprint, fit_normalizer
from .uvip_labels import score_windows
from .uvip_lock import RunLease
from .uvip_model import RealUvipPolicy
from .uvip_training import (atomic_checkpoint, load_policy, make_optimizer, move_batch,
                            restore_rng, rng_state, source_fingerprint, update_ema)


def run_preflight(config: dict) -> None:
    """Measure real Teacher/Student updates, labels, EMA and checkpoint recovery."""
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU preflight requires the authorized interactive Slurm GPU")
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    root = Path(config["output_root"]) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    lease = RunLease(root / "run.lease")
    receipt = Path(config["output_root"]) / "preflight.json"
    if receipt.exists():
        receipt.replace(root / f"previous_receipt_{time.time_ns()}.json")
    telemetry = subprocess.check_output(["nvidia-smi"], text=True)
    (root / "gpu_inventory.txt").write_text(telemetry)
    print(telemetry, flush=True)
    monitor_log = (root / "gpu_samples.csv").open("w")
    visible_uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().splitlines()
    assigned_gpu = (visible_uuids[0].strip() if len(visible_uuids) == 1
                    else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    monitor = subprocess.Popen(["nvidia-smi", "--id=" + assigned_gpu,
                                "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory",
                                "--format=csv,noheader", "--loop=1"], stdout=monitor_log, stderr=subprocess.STDOUT)
    started = time.perf_counter()
    try:
        dataset = UvipDataset(config, "train")
        options = dict(batch_size=config["batch_size"], num_workers=config["workers"], pin_memory=True,
                       shuffle=True, generator=torch.Generator().manual_seed(config["seed"]))
        if config["workers"]:
            options.update(prefetch_factor=1, persistent_workers=True)
        loader = torch.utils.data.DataLoader(dataset, **options)
        iterator = iter(loader)
        rows = []

        def batch():
            before = time.perf_counter()
            value = next(iterator)
            wait = time.perf_counter() - before
            value = move_batch(value, "cuda")
            torch.cuda.synchronize()
            return value, wait, time.perf_counter() - before

        def updates(policy, ema, optimizer, phase):
            policy.set_phase(phase)
            policy.train()
            measurements = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(6):
                value, wait, transfer = batch()
                if policy.student:
                    # Real labels exercise shape/gradient paths; these repeated
                    # smoke targets are never published as a training bank.
                    value["utility"] = torch.from_numpy(np.repeat(label["utility"].astype(np.float32),
                                                                  len(value["action"]), axis=0)).cuda()
                before = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses = policy.losses(value, phase)
                if not all(torch.isfinite(item).all() for item in losses.values()):
                    raise FloatingPointError("Nonfinite preflight loss")
                losses["loss"].backward()
                encoder_grad = any(p.grad is not None for p in policy.encoder.parameters())
                if encoder_grad != (phase in ("teacher", "C")):
                    raise ValueError("Encoder phase freeze/unfreeze failed")
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                update_ema(ema, policy, .9999)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - before
                record = dict(phase=phase, step=step, loss=losses["loss"].item(),
                              action_loss=losses["action_loss"].item(), utility_loss=losses["utility_loss"].item(),
                              grad_norm=norm.item(), data_wait_seconds=wait, data_and_transfer_seconds=transfer,
                              compute_seconds=seconds, batch_indices=value["index"].cpu().tolist())
                measurements.append(record)
                print(json.dumps(dict(event="preflight_update", **record)), flush=True)
                with (root / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            stable = measurements[2:]
            rows.append(dict(phase=phase, steps=6, finite=True,
                             mean_compute_seconds=float(np.mean([r["compute_seconds"] for r in stable])),
                             mean_data_wait_seconds=float(np.mean([r["data_wait_seconds"] for r in stable])),
                             examples_per_second=config["batch_size"] / float(np.mean([
                                 r["compute_seconds"] + r["data_and_transfer_seconds"] for r in stable])),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                             final_loss=measurements[-1]["loss"]))
            return value

        teacher = RealUvipPolicy(config).cuda()
        teacher.normalizer.fit(fit_normalizer(config))
        ema = copy.deepcopy(teacher).eval().requires_grad_(False)
        optimizer = make_optimizer(teacher, config)
        value = updates(teacher, ema, optimizer, "teacher")
        del optimizer, value
        teacher.zero_grad(set_to_none=True)
        del teacher
        teacher = ema
        teacher.eval().requires_grad_(False)
        before = time.perf_counter()
        atomic_checkpoint(root / "teacher_smoke.pt", dict(config=config, student=False, ema=teacher.state_dict()))
        teacher_write = time.perf_counter() - before
        one = torch.utils.data.default_collate([dataset[len(dataset) // 2]])
        window = dataset.index[len(dataset) // 2:len(dataset) // 2 + 1]
        before = time.perf_counter()
        label = score_windows(teacher, one, window, "train", config["label_microbatch"])
        torch.cuda.synchronize()
        label_seconds = time.perf_counter() - before
        if not all(np.isfinite(item).all() for item in label.values()):
            raise FloatingPointError("Nonfinite paired K64 labels")
        if label["utility"].shape != (1, len(config["utility_names"])):
            raise ValueError("Utility output width differs")
        np.savez(root / "label_smoke.npz", **label)
        print(json.dumps(dict(event="preflight_labels", seconds=label_seconds, utilities=label["utility"].tolist())), flush=True)
        student = RealUvipPolicy(config, student=True).cuda()
        fresh_parameter = next(student.model.parameters()).detach().clone()
        student.initialize_from_teacher(teacher, torch.ones(len(config["utility_names"]), device="cuda"))
        torch.testing.assert_close(next(student.model.parameters()), fresh_parameter, rtol=0, atol=0)
        del fresh_parameter
        for key, item in teacher.encoder.state_dict().items():
            torch.testing.assert_close(student.encoder.state_dict()[key], item, rtol=0, atol=0)
        del teacher, ema
        gc.collect()
        torch.cuda.empty_cache()
        ema = copy.deepcopy(student).eval().requires_grad_(False)
        optimizer = make_optimizer(student, config)
        for phase in ("A", "B", "C"):
            value = updates(student, ema, optimizer, phase)
        # Verify the two detach routes directly on a real GPU batch.
        optimizer.zero_grad(set_to_none=True)
        student.losses(value, "C")["utility_loss"].backward()
        if any(p.grad is not None for p in student.encoder.parameters()):
            raise ValueError("Utility gradients reached the encoder")
        optimizer.zero_grad(set_to_none=True)
        student.losses(value, "C")["action_loss"].backward()
        if any(p.grad is not None for p in student.utility_predictor.parameters()):
            raise ValueError("Action gradients reached the utility predictor")
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            expected_condition = ema.encode(move_batch(one, "cuda")["obs"])["conditioning"].clone()
        before = time.perf_counter()
        atomic_checkpoint(root / "student_resume.pt", dict(config=config, student=True, model=student.state_dict(),
                                                           ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                                                           rng=rng_state(), epoch=6, batch_cursor=6, step=18))
        resume_write = time.perf_counter() - before
        del optimizer, student, ema
        gc.collect()
        torch.cuda.empty_cache()
        before = time.perf_counter()
        restored, saved = load_policy(root / "student_resume.pt", config, "cuda")
        with torch.no_grad():
            actual_condition = restored.encode(move_batch(one, "cuda")["obs"])["conditioning"]
        torch.testing.assert_close(actual_condition, expected_condition, rtol=0, atol=0)
        prediction = restored.predict_action(move_batch(one, "cuda")["obs"], generator=torch.Generator(device="cuda").manual_seed(42))["action"]
        if prediction.shape != (1, config["n_action_steps"], config["action_dim"]) or not torch.isfinite(prediction).all():
            raise ValueError("Standalone inference shape/finite check failed")
        restored.load_state_dict(saved["model"], strict=True)
        restored.set_phase("C")
        restored.train()
        optimizer = make_optimizer(restored, config)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        assert (saved["epoch"], saved["batch_cursor"], saved["step"]) == (6, 6, 18)
        optimizer.zero_grad(set_to_none=True)
        losses = restored.losses(value, "C")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(restored.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if not torch.isfinite(losses["loss"]):
            raise ValueError("Resumed loss is nonfinite")
        restore_seconds = time.perf_counter() - before
        # Keep the large resume smoke file as a reviewable artifact, separate
        # from all formal training checkpoint/selection directories.
        atomic_json(receipt, dict(passed=True, identity=dict(config=fingerprint(config), source=source_fingerprint(config)),
                                  task=config["task"], results=rows, elapsed_seconds=time.perf_counter() - started,
                                  batch_size=config["batch_size"], workers=config["workers"],
                                  cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
                                  gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                                  gpu_job=os.environ["SLURM_JOB_ID"], gpu_node=os.environ.get("SLURMD_NODENAME"),
                                  assigned_gpu=assigned_gpu,
                                  fp32=True, amp=False, tf32=False, label_k=64, label_seconds=label_seconds,
                                  teacher_checkpoint_write_seconds=teacher_write, resume_checkpoint_write_seconds=resume_write,
                                  checkpoint_restore_and_update_seconds=restore_seconds,
                                  resume_cursor_verified=True, resume_update_passed=True, encoder_inheritance_exact=True,
                                  fresh_action_unet=True, gradient_routes_verified=True,
                                  ema_roundtrip_exact=True, inference_action_shape=list(prediction.shape),
                                  checkpoint_sha256=file_hash(root / "student_resume.pt")))
        print(json.dumps(dict(event="preflight_passed", task=config["task"], receipt=str(receipt))), flush=True)
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_log.close()
        lease.close()

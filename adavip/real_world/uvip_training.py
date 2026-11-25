"""Reproducible Teacher/Student training and self-contained EMA checkpoints."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .uvip_data import UvipDataset, atomic_json, file_hash, fingerprint, fit_normalizer
from .uvip_labels import validate_bank
from .uvip_model import RealUvipPolicy


def move_batch(value, device):
    """Move nested observations and targets without changing image dtype."""
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value


def source_fingerprint(config: dict) -> str:
    """Identify implementation, U-Net, split, source map and frozen text."""
    files = sorted(Path(__file__).parent.glob("uvip_*.py"))
    files += [Path(config["upstream_unet"]), Path(config["language_embedding"])]
    if "wipe_v2_hdf5" in config:
        repository = Path(__file__).resolve().parents[2]
        files += [repository / "scripts/real_world/run_uvip.py"]
        if config["wipe_v2_hdf5"] == "gelsight-tar":
            files += [Path(__file__).parent / "wipe_v2/gelsight.py", Path(config["raw_root"]) / "manifest.json"]
    else:
        files += [Path(config["upstream_unet"]).parents[1] / "scripts/real_world/run_uvip.py"]
    files += [Path(config["dataset_root"]) / name for name in ("dataset_manifest.json", "split.json")]
    if "teacher_reference_yaml" in config:
        files.append(Path(config["teacher_reference_yaml"]))
    return fingerprint({path.name: file_hash(path) for path in files})


def atomic_checkpoint(path: Path, payload: dict) -> None:
    """Save a complete checkpoint before replacing the last known good file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rng_state() -> dict:
    """Capture data-independent random streams for resuming a batch cursor."""
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state: dict) -> None:
    """Restore streams after constructing modules and loading weights."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def update_ema(ema, policy, decay: float) -> None:
    """Average trainable parameters; copy frozen parameters and all buffers exactly."""
    for target, value in zip(ema.parameters(), policy.parameters()):
        if value.requires_grad:
            target.lerp_(value, 1 - decay)
        else:
            target.copy_(value)
    for target, value in zip(ema.buffers(), policy.buffers()):
        target.copy_(value)


def load_policy(path: Path, config: dict | None = None, device="cpu"):
    """Restore EMA policy, normalizer and utility scales without Teacher/labels."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    policy = RealUvipPolicy(config or saved["config"], student=saved["student"])
    policy.load_state_dict(saved["ema"], strict=True)
    return policy.to(device).eval(), saved


def selected_teacher(config: dict) -> Path:
    """Accept only an explicitly selected EMA from a completed Teacher run."""
    root = Path(config["output_root"])
    selection = json.loads((root / "teacher_selection.json").read_text())
    path = Path(selection["checkpoint"])
    if selection["sha256"] != file_hash(path) or selection["source"] != source_fingerprint(config):
        raise ValueError("Selected Teacher or source contract changed")
    if selection["config"] != fingerprint(config):
        raise ValueError("Selected Teacher task config differs")
    return path


def phase_for_epoch(epoch: int, student: bool) -> str:
    """Use reference Student A=E1, B=E2..6, C=E7 onward."""
    return ("A" if epoch == 0 else "B" if epoch < 6 else "C") if student else "teacher"


def load_teacher_control(config: dict, identity: dict) -> dict | None:
    """Read an audited continuation that only shortens the Teacher run."""
    path = Path(config["output_root"]) / "teacher_control.json"
    if not path.exists():
        return None
    control = json.loads(path.read_text())
    limit = control.get("stop_after_epoch")
    if (control.get("identity") != identity
            or control.get("previous_identity", {}).get("config") != identity["config"]
            or control.get("scope") != "teacher_stop_limit_and_last_checkpoint"
            or control.get("schedule_epochs") != config["teacher_epochs"]
            or type(limit) is not int or not 1 <= limit <= config["teacher_epochs"]):
        raise ValueError("Teacher continuation control differs from its audited identity/budget")
    return control


def training_identity_matches(candidate: dict, identity: dict, control: dict | None) -> bool:
    """Allow the current identity or the explicitly recorded parent continuation."""
    return candidate == identity or (control is not None and candidate == control["previous_identity"])


def make_optimizer(policy, config: dict):
    """Include currently frozen encoder/action parameters for later A/B/C stages."""
    encoder = list(policy.encoder.parameters())
    utility = list(policy.utility_predictor.parameters()) if policy.student else []
    used = {id(value) for value in encoder + utility}
    action = [value for value in policy.parameters() if id(value) not in used]
    groups = [dict(params=encoder, name="encoder"), dict(params=action, name="action")]
    if utility:
        groups.append(dict(params=utility, name="utility"))
    return torch.optim.AdamW(groups, lr=config["learning_rate"], betas=(.95, .999),
                             weight_decay=1e-6, eps=1e-8)


def teacher_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Match the repository L1 workspace's linear warmup then cosine schedule."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return .5 * (1 + math.cos(math.pi * min(1., progress)))


def teacher_ema_decay(step: int) -> float:
    """Match EMAModel(update_after_step=0, inv_gamma=1, power=.75)."""
    adjusted = max(0, step - 1)
    return min(.9999, 1 - (1 + adjusted) ** (-.75))


def teacher_epoch_indices(config: dict, size: int, epoch: int) -> list[int]:
    """Reproduce the reference balanced sampler for a single Lego task."""
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    if config.get("teacher_sampling") == "task_balanced_with_replacement":
        weights = torch.full((size,), 1 / size, dtype=torch.double)
        return torch.multinomial(weights, size, True, generator=generator).tolist()
    return torch.randperm(size, generator=generator).tolist()


def validate_teacher_reference(config: dict) -> dict:
    """Reject unintended changes to the declared upstream teacher settings."""
    if "teacher_reference_yaml" not in config:
        return {}
    import yaml
    reference = yaml.safe_load(Path(config["teacher_reference_yaml"]).read_text())
    expected = dict(
        batch_size=reference["dataloader"]["batch_size"], workers=reference["dataloader"]["num_workers"],
        dataloader_timeout=reference["dataloader"]["timeout"], teacher_epochs=reference["training"]["num_epochs"],
        learning_rate=reference["optimizer"]["lr"], warmup_steps=reference["training"]["lr_warmup_steps"],
        seed=reference["training"]["seed"], checkpoint_steps=reference["training"]["checkpoint_every_steps"],
        checkpoint_epochs=reference["training"]["checkpoint_epochs"], horizon=reference["horizon"],
        n_action_steps=reference["n_action_steps"], down_dims=reference["policy"]["down_dims"],
        inference_steps=reference["policy"]["num_inference_steps"],
        teacher_head_seed=reference["policy"]["obs_encoder"]["head_seed"],
        history_offsets=list(range(1 - reference["n_obs_steps"], 1)), action_start=reference["n_obs_steps"] - 1,
        teacher_protocol="upstream_l1_v1", teacher_sampling="task_balanced_with_replacement",
        teacher_training_noise_seed=reference["training"]["seed"] + 300000,
    )
    errors = [key for key, value in expected.items() if config.get(key) != value]
    if errors:
        raise ValueError("L1 teacher reference mismatch: " + ", ".join(errors))
    return dict(matched=True, reference_sha256=file_hash(config["teacher_reference_yaml"]), settings=expected)


@torch.no_grad()
def validate(policy, loader, phase: str, device, seed: int) -> dict:
    """Use repeatable held-out epsilon loss without consuming training RNG."""
    policy.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals, count = {}, 0
    for batch in loader:
        batch = move_batch(batch, device)
        size = len(batch["action"])
        for key, value in policy.losses(batch, phase, generator).items():
            totals[key] = totals.get(key, 0.) + value.item() * size
        count += size
    return {key: value / count for key, value in totals.items()}


def train(config: dict, student: bool = False, resume: bool = False) -> None:
    """Train with EMA, independent validation, atomic resume and explicit provenance."""
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal training requires a Slurm GPU step")
    if not student:
        validate_teacher_reference(config)
    receipt = Path(config["output_root"]) / "preflight.json"
    teacher_receipt = Path(config["output_root"]) / "teacher_preflight.json"
    if not student and teacher_receipt.exists():
        receipt = teacher_receipt
    preflight = json.loads(receipt.read_text())
    if student and preflight.get("scope") == "teacher_only":
        raise ValueError("Teacher-only preflight cannot authorize Student training")
    identity = dict(config=fingerprint(config), source=source_fingerprint(config))
    control = None if student else load_teacher_control(config, identity)
    if not training_identity_matches(preflight.get("identity"), identity, control) or not preflight.get("passed"):
        raise ValueError("Run bounded GPU preflight for this exact task/source first")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    run = Path(config["output_root"]) / ("student" if student else "teacher")
    run.mkdir(parents=True, exist_ok=True)
    from .uvip_lock import RunLease
    lock = RunLease(run / "run.lease")
    latest = run / "latest.pt"
    if latest.exists() != resume:
        raise ValueError("Existing run requires --resume; --resume requires latest.pt")
    policy = RealUvipPolicy(config, student=student)
    initialization = {}
    labels = None
    if student:
        teacher_path = selected_teacher(config)
        labels = Path(config["output_root"]) / "labels"
        for split in ("train", "val"):
            validate_bank(labels / split, file_hash(teacher_path))
            bank = json.loads((labels / split / "manifest.json").read_text())["identity"]
            if bank["source"] != identity["source"] or bank["config"] != identity["config"]:
                raise ValueError("Label source/config differs")
        teacher, _ = load_policy(teacher_path, config)
        policy.initialize_from_teacher(teacher, torch.from_numpy(np.load(labels / "train/rms.npy")))
        initialization = dict(teacher_sha256=file_hash(teacher_path),
                              labels={s: file_hash(labels / s / "COMPLETE.json") for s in ("train", "val")})
        del teacher
    else:
        policy.normalizer.fit(fit_normalizer(config))
    policy.to(device)
    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = make_optimizer(policy, config)
    datasets = {s: UvipDataset(config, s, labels / s if labels else None) for s in ("train", "val")}
    batch_size = config["batch_size"]
    steps_per_epoch = math.ceil(len(datasets["train"]) / batch_size)
    epochs = config["student_epochs" if student else "teacher_epochs"]
    total_steps = epochs * steps_per_epoch
    stop_epoch = control["stop_after_epoch"] if control else epochs
    workers = config["workers"]
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=1, timeout=config.get("dataloader_timeout", 0))
    validation = DataLoader(datasets["val"], shuffle=False,
                            generator=torch.Generator().manual_seed(123), **options)
    epoch, cursor, step, best = 0, 0, 0, float("inf")
    metrics = None
    noise_generator = None
    if not student and "teacher_training_noise_seed" in config:
        noise_generator = torch.Generator(device=device).manual_seed(config["teacher_training_noise_seed"])
    if resume:
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if not training_identity_matches(saved["identity"], identity, control) or saved["initialization"] != initialization:
            raise ValueError("Checkpoint source/config/Teacher/labels changed")
        policy.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, step, best = (saved[key] for key in ("epoch", "batch_cursor", "step", "best"))
        metrics = saved.get("metrics")
        restore_rng(saved["rng"])
        if noise_generator is not None:
            noise_generator.set_state(saved["training_noise_rng"])
    if epoch > stop_epoch or (epoch == stop_epoch and cursor):
        raise ValueError("Checkpoint has already advanced beyond the requested stop epoch")
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    def save(path, metrics=None):
        atomic_checkpoint(path, dict(config=config, identity=identity, initialization=initialization,
                                    student=student, model=policy.state_dict(), ema=ema.state_dict(),
                                    optimizer=optimizer.state_dict(), epoch=epoch, batch_cursor=cursor,
                                    step=step, best=best, rng=rng_state(), metrics=metrics,
                                    training_noise_rng=noise_generator.get_state() if noise_generator is not None else None,
                                    training_control=control))
    while epoch < stop_epoch:
        phase = phase_for_epoch(epoch, student)
        policy.set_phase(phase)
        policy.train()
        permutation = (teacher_epoch_indices(config, len(datasets["train"]), epoch) if not student else
                       torch.randperm(len(datasets["train"]), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist())
        remaining = permutation[cursor * batch_size:]
        loader = DataLoader(datasets["train"], sampler=remaining,
                            generator=torch.Generator().manual_seed(epoch + 987), **options)
        for batch in loader:
            factor = min(1., (step + 1) / config["warmup_steps"]) * .5 * (1 + math.cos(math.pi * step / total_steps))
            if not student and config.get("teacher_protocol") == "upstream_l1_v1":
                factor = teacher_lr_multiplier(step, total_steps, config["warmup_steps"])
            for group in optimizer.param_groups:
                ramp = min(1., max(0, step - 6 * steps_per_epoch + 1) / 500) if student and group["name"] == "encoder" else 1.
                group["lr"] = config["learning_rate"] * factor * ramp
            optimizer.zero_grad(set_to_none=True)
            value = move_batch(batch, device)
            losses = (policy.losses(value, phase, noise_generator) if noise_generator is not None else
                      policy.losses(value, phase))
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite training loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            decay = .9999 if student else min(.9999, 1 - (1 + step) ** (-.75))
            if not student and config.get("teacher_protocol") == "upstream_l1_v1":
                decay = teacher_ema_decay(step)
            update_ema(ema, policy, decay)
            cursor, step = cursor + 1, step + 1
            if step == 1 or step % 20 == 0:
                print(json.dumps(dict(event="train", epoch=epoch + 1, phase=phase, step=step,
                                      losses={k: v.item() for k, v in losses.items()})), flush=True)
            if step % config["checkpoint_steps"] == 0 or stopping[0]:
                save(latest)
            if stopping[0]:
                return
        metrics = validate(ema, validation, phase, device, config["seed"] + 10000)
        # A-stage utility-only checkpoints cannot win against action-trained stages.
        score = metrics["loss"] if phase != "A" else float("inf")
        improved = score < best
        best = min(best, score)
        epoch, cursor = epoch + 1, 0
        save(latest, metrics)
        if improved:
            save(run / "best.pt", metrics)
        if epoch in config.get("checkpoint_epochs", (1, 6, 50, 100, 200, 300, 400, 500, 600)):
            save(run / f"epoch_{epoch:04d}.pt", metrics)
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(epoch=epoch, step=step, phase=phase, validation=metrics)) + "\n")
        print(json.dumps(dict(event="epoch", epoch=epoch, validation=metrics)), flush=True)
    save(run / "last.pt", metrics)
    atomic_json(run / "COMPLETE.json", dict(identity=identity, epochs=epoch, best=best,
                                            schedule_epochs=epochs, stop_after_epoch=stop_epoch,
                                            training_control=control,
                                            last_checkpoint_sha256=file_hash(run / "last.pt"),
                                            checkpoint_sha256=file_hash(run / "best.pt")))
    if not student:
        atomic_json(Path(config["output_root"]) / "teacher_selection.json",
                    dict(checkpoint=str(run / "best.pt"), sha256=file_hash(run / "best.pt"),
                         selection="minimum held-out EMA epsilon MSE", **identity))

"""Causal, episode-split real-robot RGB/GelSight data for DP + UViP."""

from __future__ import annotations

import hashlib
import concurrent.futures
import json
import os
from pathlib import Path
import tarfile
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


def file_hash(path: str | Path) -> str:
    """Hash a persistent file without loading it into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    """Hash JSON-compatible input contracts with deterministic serialization."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish a JSON file only after its contents have been flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dataset_sources(config: dict) -> list[dict]:
    """Read episode ranges and translate the Wipe V2 publisher manifest."""
    root = Path(config["dataset_root"])
    sources = json.loads((root / "dataset_manifest.json").read_text())["sources"]
    layout = config.get("wipe_v2_hdf5")
    if layout is None:
        return sources
    if layout not in ("raw", "cleaned", "gelsight-tar"):
        raise ValueError(f"Unknown Wipe V2 image layout: {layout}")
    archives = {}
    if layout == "gelsight-tar":
        from .wipe_v2.gelsight import match_archives

        archives = match_archives(Path(config["raw_root"]), sources)
    store = zarr.open_group(str(root / "replay_buffer.zarr"), mode="r")
    ids = store["meta/episode_ids"][:].tolist()
    ends = store["meta/episode_ends"][:].tolist()
    by_id = {source["episode_id"]: source for source in sources}
    if len(by_id) != len(sources) or len(ids) != len(ends) or len(ids) != len(sources) or set(ids) != set(by_id):
        raise ValueError("Wipe V2 manifest and Zarr episode IDs disagree")
    normalized = []
    for index, (episode_id, start, end) in enumerate(zip(ids, [0] + ends, ends)):
        source = by_id[episode_id]
        if (start, end, end - start) != (source["start"], source["end"], source["frames"]):
            raise ValueError(f"Wipe V2 row range disagrees for episode {episode_id}")
        first, stop = source["retained_start"], source["retained_stop"]
        if not (0 <= first < stop <= source["raw_frames"] and stop - first == end - start):
            raise ValueError(f"Invalid retained slice for episode {episode_id}")
        entry = dict(
            file=f"episode_{episode_id}.hdf5", zarr_episode_index=index,
            start=start, end=end, frames=end-start, row_offset=first if layout == "raw" else 0,
            image_rows=source["raw_frames"] if layout == "raw" else end-start,
            sha256=source["raw_sha256"] if layout == "raw" else source["source_sha256"],
        )
        if archives:
            record = archives[episode_id]
            entry.update(file=record["path"], sha256=record["sha256"], gelsight_archive=record)
        normalized.append(entry)
    return normalized


def make_index(config: dict, split: str) -> np.ndarray:
    """Enumerate every original anchor once; columns are episode, local frame."""
    if split not in ("train", "val"):
        raise ValueError("Unknown split")
    root = Path(config["dataset_root"])
    partition = json.loads((root / "split.json").read_text())
    selected = set(partition["train_episode_ids" if split == "train" else "validation_episode_ids"])
    rows = []
    for source in dataset_sources(config):
        episode_id = int(Path(source["file"]).stem.split("_")[-1])
        if episode_id in selected:
            rows.extend((source["zarr_episode_index"], frame) for frame in range(source["frames"]))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def fit_normalizer(config: dict) -> dict[str, list]:
    """Fit min/max only to original train rows; retain reversible constants."""
    root = Path(config["dataset_root"])
    partition = json.loads((root / "split.json").read_text())
    train = set(partition["train_episode_ids"])
    store = zarr.open_group(str(root / "replay_buffer.zarr"), mode="r")
    result = {}
    sources = dataset_sources(config)
    for output, key in (("state", config["state_key"]), ("action", "action")):
        values = np.concatenate([
            store["data/" + key][source["start"]:source["end"]]
            for source in sources
            if int(Path(source["file"]).stem.split("_")[-1]) in train
        ]).astype(np.float64)
        if values.ndim != 2 or values.shape[1] != config["action_dim"] or not np.isfinite(values).all():
            raise ValueError("Joint width/finite-value contract violated")
        minimum, maximum = values.min(0), values.max(0)
        span = maximum - minimum
        scale = np.where(span > 1e-7, 2 / np.maximum(span, 1e-7), 1)
        offset = np.where(span > 1e-7, -1 - scale * minimum, -minimum)
        for name, array in (("scale", scale), ("offset", offset), ("min", minimum), ("max", maximum)):
            result[output + "_" + name] = array.astype(np.float32).tolist()
    return result


def decode_rgb(file: h5py.File, key: str, frame: int, order: list[str]) -> np.ndarray:
    """Decode the recorded JPEG length and explicitly convert OpenCV BGR to RGB."""
    length = int(file["compress_len"][order.index(key), frame])
    encoded = file["observations/images/" + key][frame, :length]
    return _decode_jpeg(encoded, f"{file.filename}/{key}/{frame}")


def _decode_jpeg(encoded: np.ndarray, location: str) -> np.ndarray:
    """Use the same RGB conversion for bulk cache and direct frame reads."""
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.shape[-1] != 3:
        raise ValueError(f"Cannot decode {location}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _tar_images(config: dict, source: dict, path: Path):
    """Yield one resized stream at a time from the tar or the aligned Zarr RGB."""
    from .wipe_v2.gelsight import GELSIGHT_KEYS, cleaned_frame_map, read_member

    record = source["gelsight_archive"]
    store = zarr.open_group(str(Path(config["dataset_root"]) / "replay_buffer.zarr"), mode="r")
    with tarfile.open(path, "r:") as archive:
        for key in config["image_keys"]:
            height, width = config["image_shapes"][key]
            array = np.empty((source["frames"], height, width, 3), dtype=np.uint8)
            if key in GELSIGHT_KEYS:
                members = cleaned_frame_map(archive, record, key)
                images = (
                    _decode_jpeg(np.frombuffer(read_member(archive, name), dtype=np.uint8), f"{path}:{name}")
                    for name in members
                )
            else:
                rgb = store["data/" + key][source["start"]:source["end"]]
                if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) != len(array):
                    raise ValueError(f"Invalid Wipe V2 RGB stream: {key}")
                images = iter(rgb)
            for frame, image in enumerate(images):
                array[frame] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            yield key, array


def _hdf5_images(config: dict, source: dict, path: Path):
    """Yield one resized stream at a time from a full HDF5 recording."""
    with h5py.File(path, "r") as file:
        for key in config["image_keys"]:
            height, width = config["image_shapes"][key]
            images = file["observations/images/" + key]
            if "image_rows" in source:
                if images.ndim != 2 or images.dtype != np.uint8 or len(images) != source["image_rows"]:
                    raise ValueError(f"Unexpected Wipe V2 JPEG rows: {path}/{key}")
                start = source["row_offset"]
                encoded = images[start:start + source["frames"]]
                lengths = np.full(len(encoded), encoded.shape[1], dtype=np.int64)
            else:
                lengths = file["compress_len"][config["raw_image_order"].index(key), :]
                encoded = images[:]
            array = np.empty((source["frames"], height, width, 3), dtype=np.uint8)
            for frame in range(source["frames"]):
                image = _decode_jpeg(encoded[frame, :int(lengths[frame])], f"{path}/{key}/{frame}")
                array[frame] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            yield key, array


def _cache_episodes(config: dict, output: Path, sources: list, contract: str):
    """Process disjoint episodes; each completion is independently resumable."""
    for source in sources:
        path = Path(config["raw_root"]) / source["file"]
        episode = output / f"episode_{source['zarr_episode_index']:04d}"
        episode.mkdir(exist_ok=True)
        done = episode / "COMPLETE.json"
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["cache_contract"] != contract or saved["source_sha256"] != source["sha256"]:
                raise ValueError("Episode cache identity changed")
            for key in config["image_keys"]:
                array = np.load(episode / (key + ".npy"), mmap_mode="r")
                if array.shape != (source["frames"], *config["image_shapes"][key], 3) or array.dtype != np.uint8:
                    raise ValueError("Episode cache incomplete")
                if file_hash(episode / (key + ".npy")) != saved["files"][key]:
                    raise ValueError("Cached image checksum mismatch")
            yield source["file"]
            continue
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"Image archive hash differs: {path}")
        reader = _tar_images if "gelsight_archive" in source else _hdf5_images
        for key, array in reader(config, source, path):
            temporary = episode / (key + ".npy.partial")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, episode / (key + ".npy"))
        atomic_json(done, dict(cache_contract=contract, source_sha256=source["sha256"],
                               files={key: file_hash(episode / (key + ".npy")) for key in config["image_keys"]}))
        yield source["file"]


def cache_identity(config: dict) -> dict:
    """Identify the dataset, source layout and image transformation."""
    root = Path(config["dataset_root"])
    identity = dict(dataset_manifest=file_hash(root / "dataset_manifest.json"),
                    images=config["image_keys"], shapes=config["image_shapes"],
                    image_order=config["raw_image_order"], color="RGB", interpolation="INTER_LINEAR")
    if "wipe_v2_hdf5" in config:
        identity["wipe_v2_hdf5"] = config["wipe_v2_hdf5"]
    if config.get("wipe_v2_hdf5") == "gelsight-tar":
        identity["gelsight_manifest"] = file_hash(Path(config["raw_root"]) / "manifest.json")
    return identity


def build_image_cache(config: dict, output: Path, stop_after: int = 0) -> dict:
    """Decode RGB into resumable per-episode files using bounded CPU threads."""
    identity = cache_identity(config)
    contract = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    from .uvip_lock import RunLease
    lock = RunLease(output / "cache.lease")
    marker = output / "manifest.json"
    if marker.exists() and json.loads(marker.read_text())["identity"] != identity:
        raise ValueError("Cache contract changed; use a separate cache directory")
    atomic_json(marker, dict(identity=identity, fingerprint=contract))
    cv2.setNumThreads(1)
    all_sources = dataset_sources(config)
    sources = all_sources[:stop_after or None]
    def process(source):
        return next(_cache_episodes(config, output, [source], contract))
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.get("cache_workers", 4)) as executor:
        for file in executor.map(process, sources):
            completed.append(file)
            print(json.dumps(dict(event="cache_episode", file=file, completed=len(completed))), flush=True)
    if len(completed) != len(all_sources):
        return dict(complete=False, episodes=len(completed))
    result = dict(complete=True, episodes=len(completed), fingerprint=contract)
    atomic_json(output / "COMPLETE.json", result)
    return result


class UvipDataset(Dataset):
    """Read causal history and joint-action chunks without future tactile input."""

    def __init__(self, config: dict, split: str, labels: Path | None = None, require_cache: bool = True):
        self.config = config
        self.root = Path(config["dataset_root"])
        self.sources = dataset_sources(config)
        self.index = make_index(config, split)
        self.store = zarr.open_group(str(self.root / "replay_buffer.zarr"), mode="r")
        self.embedding = np.load(config["language_embedding"], allow_pickle=False).astype(np.float32)
        if self.embedding.shape != (512,) or not np.isfinite(self.embedding).all():
            raise ValueError("Expected finite frozen CLIP embedding [512]")
        self.cache = Path(config["cache_root"])
        self.require_cache = require_cache
        self._images: dict = {}
        self._pid = os.getpid()
        if require_cache and not (self.cache / "COMPLETE.json").exists():
            raise ValueError("Image cache has not completed")
        if require_cache:
            saved = json.loads((self.cache / "manifest.json").read_text())
            expected = cache_identity(config)
            complete = json.loads((self.cache / "COMPLETE.json").read_text())
            if saved["identity"] != expected or complete["fingerprint"] != fingerprint(expected):
                raise ValueError("Image cache source/resize contract differs")
        self.labels = None if labels is None else np.load(labels / "utility.npy", mmap_mode="r")
        if labels is not None and not np.array_equal(np.load(labels / "index.npy"), self.index):
            raise ValueError("Utility/window join mismatch")

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_images"] = {}
        return state

    def _image(self, episode: int, key: str, indices: np.ndarray) -> np.ndarray:
        if self._pid != os.getpid():
            self._images, self._pid = {}, os.getpid()
        path = self.cache / f"episode_{episode:04d}" / (key + ".npy")
        if path.exists():
            # Limit worker descriptors while retaining useful page-cache locality.
            if (episode, key) not in self._images:
                if len(self._images) >= 20:
                    self._images.clear()
                self._images[episode, key] = np.load(path, mmap_mode="r")
            return np.asarray(self._images[episode, key][indices])
        if self.require_cache or "wipe_v2_hdf5" in self.config:
            raise FileNotFoundError(path)
        height, width = self.config["image_shapes"][key]
        with h5py.File(Path(self.config["raw_root"]) / self.sources[episode]["file"], "r") as file:
            return np.stack([cv2.resize(decode_rgb(file, key, int(i), self.config["raw_image_order"]),
                                       (width, height), interpolation=cv2.INTER_LINEAR) for i in indices])

    def __getitem__(self, row: int) -> dict:
        episode, anchor = map(int, self.index[row])
        source = self.sources[episode]
        history = np.asarray(self.config["history_offsets"]) + anchor
        if history.max() > anchor:
            raise ValueError("Future observations are forbidden")
        history = history.clip(0, source["frames"] - 1)
        actions = np.arange(self.config["horizon"]) + anchor - self.config["action_start"]
        valid = (actions >= 0) & (actions < source["frames"])
        action_rows = actions.clip(0, source["frames"] - 1) + source["start"]
        obs = {key: torch.from_numpy(np.ascontiguousarray(np.moveaxis(self._image(episode, key, history), -1, 1)))
               for key in self.config["image_keys"]}
        obs["state"] = torch.from_numpy(self.store["data/" + self.config["state_key"]].oindex[history + source["start"]])
        obs["language"] = torch.from_numpy(self.embedding.copy())
        item = dict(obs=obs, action=torch.from_numpy(self.store["data/action"].oindex[action_rows]),
                    action_mask=torch.from_numpy(np.broadcast_to(valid[:, None], (len(actions), self.config["action_dim"])).copy()),
                    index=torch.tensor(row))
        if self.labels is not None:
            item["utility"] = torch.from_numpy(np.asarray(self.labels[row], dtype=np.float32).copy())
        return item

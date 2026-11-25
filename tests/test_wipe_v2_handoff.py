"""Cover the Wipe V2 config and the publisher's HDF5-to-Zarr row alignment."""

import json
import io
import os
from pathlib import Path
import tarfile

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")
zarr = pytest.importorskip("zarr")
pytest.importorskip("torch")

from adavip.real_world.uvip_data import UvipDataset, build_image_cache, file_hash, fit_normalizer


@pytest.mark.parametrize("layout", ["raw", "cleaned"])
def test_uvip_retains_publisher_rows_and_actions(tmp_path, layout, monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr("adavip.real_world.uvip_lock.process_birth", lambda pid: str(pid))
    data = tmp_path / "data"
    data.mkdir()
    store = zarr.open_group(str(data / "replay_buffer.zarr"), mode="w")
    store["meta/episode_ids"] = [8, 13]
    store["meta/episode_ends"] = [2, 4]
    store["data/action"] = np.repeat(np.array([[10], [20], [900], [999]], np.float32), 14, axis=1)
    store["data/robot_qpos"] = store["data/action"][:]
    sources = []
    camera = "gelsight_left"
    for index, episode in enumerate((8, 13)):
        source = tmp_path / f"episode_{episode}.hdf5"
        values = [3, 70, 140, 210] if layout == "raw" else [70, 140]
        frames = [cv2.imencode(".jpg", np.full((8, 8, 3), value, np.uint8))[1] for value in values]
        rows = np.zeros((len(frames), 1024), np.uint8)
        for row, frame in zip(rows, frames):
            row[:len(frame)] = frame
        with h5py.File(source, "w") as handle:
            handle[f"observations/images/{camera}"] = rows
        sources.append(dict(episode_id=episode, start=index * 2, end=index * 2 + 2, frames=2,
                            retained_start=1, retained_stop=3, raw_frames=4,
                            raw_sha256=file_hash(source), source_sha256=file_hash(source)))
    (data / "dataset_manifest.json").write_text(json.dumps({"sources": sources[::-1]}))
    (data / "split.json").write_text(json.dumps(dict(train_episode_ids=[8], validation_episode_ids=[13])))
    np.save(tmp_path / "language.npy", np.ones(512, np.float32))
    config = dict(dataset_root=str(data), raw_root=str(tmp_path), wipe_v2_hdf5=layout,
                  image_keys=[camera], image_shapes={camera: [8, 8]}, raw_image_order=[camera],
                  language_embedding=str(tmp_path / "language.npy"), state_key="robot_qpos",
                  cache_root=str(tmp_path / "cache"), action_dim=14, history_offsets=[-2, 0],
                  horizon=32, action_start=3)
    assert build_image_cache(config, Path(config["cache_root"]))["complete"]
    train = UvipDataset(config, "train")
    heldout = UvipDataset(config, "val")
    assert len(train) == len(heldout) == 2
    assert train[1]["obs"][camera][:, 0, 0, 0].tolist() == [70, 140]
    assert train[1]["action"][3].tolist() == [20] * 14
    assert heldout[0]["action"][3].tolist() == [900] * 14
    assert fit_normalizer(config)["action_max"] == [20] * 14


def test_published_tar_and_zarr_share_cleaned_indices(tmp_path):
    store = zarr.open_group(str(tmp_path / "replay_buffer.zarr"), mode="w")
    store["meta/episode_ids"] = [8]
    store["meta/episode_ends"] = [2]
    store["data/cam_high"] = np.stack([np.full((8, 8, 3), (10, 20, 30), np.uint8)] * 2)
    sensor = "gelsight_left"
    path = tmp_path / "episode_8.tar"
    with tarfile.open(path, "w") as archive:
        def put(name, data):
            member = tarfile.TarInfo(f"episode_8/{sensor}/{name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

        put("timestamps.csv", b"raw_frame_index,cleaned_frame_index\n2,1\n0,\n1,0\n")
        for index, value in ((1, 70), (2, 140)):
            put(f"{index:06d}.jpg", cv2.imencode(".jpg", np.full((8, 8, 3), value, np.uint8))[1].tobytes())
    publisher = dict(episode_id=8, raw_frames=3, frames=2, start=0, end=2,
                     retained_start=1, retained_stop=3, raw_sha256="source", source_sha256="cleaned")
    record = dict(episode_id=8, raw_frames=3, cleaned_frames=2, retained_start=1,
                  retained_stop_exclusive=3, source_hdf5_sha256="source", path=path.name,
                  sha256=file_hash(path), sensors=["gelsight_left", "gelsight_right"])
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"sources": [publisher]}))
    (tmp_path / "manifest.json").write_text(json.dumps({"records": [record]}))
    from adavip.real_world.uvip_data import _tar_images, dataset_sources

    config = dict(dataset_root=str(tmp_path), raw_root=str(tmp_path), wipe_v2_hdf5="gelsight-tar",
                  image_keys=[sensor, "cam_high"], image_shapes={key: [8, 8] for key in (sensor, "cam_high")})
    images = dict(_tar_images(config, dataset_sources(config)[0], path))
    assert images[sensor][:, 0, 0, 0].tolist() == [70, 140]
    assert images["cam_high"][0, 0, 0].tolist() == [10, 20, 30]

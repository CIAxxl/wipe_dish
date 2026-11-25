"""Join the published GelSight JPEG archives to the cleaned Wipe V2 rows."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tarfile


GELSIGHT_KEYS = ("gelsight_left", "gelsight_right")


def match_archives(directory: Path, sources: list[dict]) -> dict[int, dict]:
    """Check that the archive release references the same captures and slices."""
    records = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["records"]
    by_id = {record["episode_id"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != {source["episode_id"] for source in sources}:
        raise ValueError("GelSight and Wipe V2 manifest episode IDs disagree")
    for source in sources:
        episode = source["episode_id"]
        record = by_id[episode]
        if (
            record["path"] != f"episode_{episode}.tar"
            or record["source_hdf5_sha256"] != source["raw_sha256"]
            or record["raw_frames"] != source["raw_frames"]
            or record["cleaned_frames"] != source["frames"]
            or record["retained_start"] != source["retained_start"]
            or record["retained_stop_exclusive"] != source["retained_stop"]
            or set(record["sensors"]) != set(GELSIGHT_KEYS)
        ):
            raise ValueError(f"GelSight provenance disagrees for episode {episode}")
    return by_id


def read_member(archive: tarfile.TarFile, name: str) -> bytes:
    """Read a regular tar member without extracting to the filesystem."""
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected regular member: {archive.name}:{name}")
    with archive.extractfile(member) as stream:
        return stream.read()


def cleaned_frame_map(archive: tarfile.TarFile, record: dict, sensor: str) -> list[str]:
    """Resolve retained JPEG names in cleaned frame order using timestamps.csv."""
    prefix = f"episode_{record['episode_id']}/{sensor}"
    timestamps = read_member(archive, f"{prefix}/timestamps.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(timestamps)))
    raw_indices = [int(row["raw_frame_index"]) for row in rows]
    if len(rows) != record["raw_frames"] or set(raw_indices) != set(range(len(rows))):
        raise ValueError(f"Invalid raw frame indices: {archive.name}:{sensor}")
    retained = sorted(
        (int(row["cleaned_frame_index"]), raw)
        for row, raw in zip(rows, raw_indices)
        if row["cleaned_frame_index"] != ""
    )
    expected = list(enumerate(range(record["retained_start"], record["retained_stop_exclusive"])))
    if retained != expected:
        raise ValueError(f"Invalid cleaned frame indices: {archive.name}:{sensor}")
    return [f"{prefix}/{raw:06d}.jpg" for _, raw in retained]

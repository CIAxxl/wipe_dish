"""Recoverable process leases for persistent shared-storage task runs."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import uuid
import weakref
from pathlib import Path

from .uvip_data import atomic_json


def process_birth(pid: int) -> str:
    """Distinguish a live Linux process from a reused PID after restart."""
    return (Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" +
            Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def release_lease(path: Path, token: str) -> None:
    """Remove only the caller's own lease directory, never another owner's."""
    try:
        if json.loads((path / "owner.json").read_text())["token"] == token:
            (path / "owner.json").unlink()
            path.rmdir()
    except FileNotFoundError:
        pass


class RunLease:
    """Acquire an atomic directory lease and recover demonstrably dead owners."""

    def __init__(self, path: Path):
        self.path = path
        for _ in range(2):
            try:
                path.mkdir()
            except FileExistsError:
                try:
                    owner = json.loads((path / "owner.json").read_text())
                except FileNotFoundError as error:
                    raise BlockingIOError("Lease owner is being published; retry shortly") from error
                if owner["host"] == socket.gethostname():
                    try:
                        alive = process_birth(owner["pid"]) == owner["birth"]
                    except FileNotFoundError:
                        alive = False
                else:
                    alive = True
                    if owner.get("job") and owner.get("step"):
                        result = subprocess.run(["scontrol", "show", "step",
                                                 owner["job"] + "." + owner["step"]],
                                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                timeout=20)
                        message = result.stdout.lower()
                        alive = not (result.returncode and ("invalid job" in message or "not found" in message))
                if alive:
                    raise BlockingIOError(f"Run already owned by {owner['host']} PID {owner['pid']}")
                release_lease(path, owner["token"])
                continue
            token = uuid.uuid4().hex
            try:
                atomic_json(path / "owner.json", dict(token=token, host=socket.gethostname(), pid=os.getpid(),
                                                       birth=process_birth(os.getpid()),
                                                       job=os.environ.get("SLURM_JOB_ID"),
                                                       step=os.environ.get("SLURM_STEP_ID")))
            except BaseException:
                path.rmdir()
                raise
            self._finalizer = weakref.finalize(self, release_lease, path, token)
            return
        raise BlockingIOError("Lease ownership changed during recovery; retry")

    def close(self) -> None:
        """Release a successfully acquired lease explicitly."""
        self._finalizer()

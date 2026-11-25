"""Exercise checkpoint retention with the native workspace serializer."""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dill = pytest.importorskip("dill")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("reactive_diffusion_policy")

from adavip.real_world.wipe_v2.at_workspace import WipeV2ATWorkspace
from adavip.real_world.wipe_v2.dp_workspace import WipeV2DPWorkspace
from adavip.real_world.wipe_v2.ldp_workspace import WipeV2LDPWorkspace
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace


def completed_epoch(path: Path) -> int:
    """Read the real epoch counter from the serialized workspace."""
    payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
    return dill.loads(payload["pickles"]["epoch"]) + 1


@pytest.mark.parametrize("workspace_type", [WipeV2DPWorkspace, WipeV2LDPWorkspace])
def test_milestones_survive_latest_and_best_updates(tmp_path, monkeypatch, workspace_type):
    monkeypatch.setattr("adavip.real_world.wipe_v2.ldp_workspace.assert_frozen", lambda *_: None)
    workspace = workspace_type.__new__(workspace_type)
    BaseWorkspace.__init__(workspace, OmegaConf.create({"checkpoint": {"keep_epochs": [20, 40, 60, 100]}}),
                           output_dir=str(tmp_path))
    workspace.frozen_at = workspace.ema_frozen_at = workspace._at_expected = None
    checkpoint_dir = tmp_path / "checkpoints"
    for epoch in (19, 20, 40, 60, 100, 101):
        workspace.epoch = epoch - 1
        workspace.save_checkpoint()
        if epoch in (19, 60):
            workspace.save_checkpoint(path=checkpoint_dir / f"epoch={epoch-1:04d}-val_loss=0.ckpt")

    assert {path.name: completed_epoch(path) for path in checkpoint_dir.glob("epoch-*.ckpt")} == {
        f"epoch-{epoch:04d}.ckpt": epoch for epoch in (20, 40, 60, 100)
    }
    assert completed_epoch(checkpoint_dir / "best.ckpt") == 60
    assert completed_epoch(checkpoint_dir / "latest.ckpt") == 101


def test_ldp_accepts_at_best_from_an_earlier_epoch(tmp_path, monkeypatch):
    at = WipeV2ATWorkspace.__new__(WipeV2ATWorkspace)
    BaseWorkspace.__init__(at, OmegaConf.create({}), output_dir=str(tmp_path))
    at.model = torch.nn.Linear(1, 1)
    checkpoint_dir = tmp_path / "checkpoints"
    at.epoch = 99
    at.save_checkpoint(path=checkpoint_dir / "epoch=0099-train_loss=0.ckpt")
    at.epoch = 600
    at.save_checkpoint()

    def initialize_policy(workspace, cfg, output_dir=None):
        workspace.model = workspace.ema_model = SimpleNamespace(at=object())

    monkeypatch.setattr(WipeV2DPWorkspace, "__init__", initialize_policy)
    monkeypatch.setattr("adavip.real_world.wipe_v2.ldp_workspace.assert_frozen", lambda *_: None)
    ldp = WipeV2LDPWorkspace(OmegaConf.create({"at_load_dir": str(checkpoint_dir / "best.ckpt")}))
    assert ldp.at_provenance["completed_epoch"] == 100
    assert completed_epoch(checkpoint_dir / "latest.ckpt") == 601

"""Regression coverage for the PCA-reserved dual-arm V2 partition."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')
zarr = pytest.importorskip('zarr')
pytest.importorskip('reactive_diffusion_policy')

MODULE = Path(os.environ.get('WIPE_V2_DATASET_MODULE', str(
    Path(__file__).resolve().parents[1] / 'adavip/real_world/wipe_v2/dataset.py')))
spec = importlib.util.spec_from_file_location('wipe_v2_dataset_test', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_dataset(path: Path) -> tuple[dict, dict]:
    """Create a nonrandom split with conspicuously out-of-range validation rows."""
    validation = [2, 5, 18, 20, 26, 28, 31, 32, 34, 40]
    train = [index for index in range(50) if index not in validation]
    split = {'seed': 42, 'train_episode_ids': train, 'validation_episode_ids': validation,
             'pca_fit_episode_ids': [3, 4], 'pca_heldout_episode_ids': [28, 32]}
    path.mkdir()
    (path / 'split.json').write_text(json.dumps(split))
    root = zarr.open_group(str(path / 'replay_buffer.zarr'), mode='w')
    data, meta = root.create_group('data'), root.create_group('meta')
    meta.array('episode_ends', np.arange(1, 51, dtype=np.int64) * 4)
    meta.array('episode_ids', np.arange(50, dtype=np.int64))
    meta.array('train_mask', np.isin(np.arange(50), train))
    meta.array('val_mask', np.isin(np.arange(50), validation))
    values = np.repeat(np.where(np.isin(np.arange(50), validation), 1000, np.arange(50)), 4)
    obs = {}
    for key, width in [('robot_qpos', 14), ('gelsight_left_marker_offset_emb', 15),
                       ('gelsight_right_marker_offset_emb', 15), ('action', 14)]:
        data.array(key, np.repeat(values[:, None], width, axis=1).astype(np.float32))
        if key != 'action':
            obs[key] = {'shape': [width], 'type': 'low_dim'}
    shape = {'obs': obs, 'extended_obs': {key: value for key, value in obs.items() if key != 'robot_qpos'},
             'action': {'shape': [14]}}
    return split, shape


def test_reserved_split_and_train_only_normalization(tmp_path: Path) -> None:
    """The saved holdout controls sampling and cannot influence fitted limits."""
    split, shape = make_dataset(tmp_path / 'dataset')
    dataset = module.WipeV2Dataset(str(tmp_path / 'dataset'), shape_meta=shape,
                                  horizon=4, pad_before=0, pad_after=0, n_obs_steps=2,
                                  val_ratio=0.2, seed=42)
    assert np.flatnonzero(dataset.val_mask).tolist() == split['validation_episode_ids']
    assert len(dataset) == 40 and len(dataset.get_validation_dataset()) == 10
    maximum = dataset.get_normalizer()['action'].state_dict()['params_dict.input_stats.max']
    assert maximum.tolist() == [49.0] * 14


def test_reject_pca_holdout_in_training(tmp_path: Path) -> None:
    """Reject a metadata-consistent partition that leaks a PCA holdout."""
    split, shape = make_dataset(tmp_path / 'dataset')
    split['pca_heldout_episode_ids'] = [3]
    (tmp_path / 'dataset/split.json').write_text(json.dumps(split))
    with pytest.raises(AssertionError):
        module.WipeV2Dataset(str(tmp_path / 'dataset'), shape_meta=shape, val_ratio=0.2, seed=42)

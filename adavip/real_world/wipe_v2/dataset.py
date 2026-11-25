"""Use the publisher's PCA-reserved V2 split with the existing Piper sampler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adavip.real_world.piper_rdp_dataset import PiperRdpDataset
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer


class WipeV2Dataset(PiperRdpDataset):
    """Validate saved episode IDs and masks without generating a random split."""

    def __init__(self, dataset_path: str, **kwargs: Any) -> None:
        path = Path(dataset_path)
        split = json.loads((path / 'split.json').read_text())
        source = ReplayBuffer.create_from_path(str(path / 'replay_buffer.zarr'), mode='r')
        ids = np.asarray(source.meta['episode_ids'])
        ends = np.asarray(source.meta['episode_ends'])
        assert len(ids) == 50 and len(set(ids.tolist())) == 50
        assert len(ends) == 50 and np.all(np.diff(np.r_[0, ends]) > 0)
        train_ids = split['train_episode_ids']
        val_ids = split['validation_episode_ids']
        assert len(train_ids) == len(set(train_ids)) == 40
        assert len(val_ids) == len(set(val_ids)) == 10
        assert set(train_ids).isdisjoint(val_ids)
        assert set(train_ids) | set(val_ids) == set(ids.tolist())
        assert set(split['pca_fit_episode_ids']) <= set(train_ids)
        assert set(split['pca_heldout_episode_ids']) <= set(val_ids)
        train_mask, val_mask = np.isin(ids, train_ids), np.isin(ids, val_ids)
        assert np.array_equal(np.asarray(source.meta['train_mask'], dtype=bool), train_mask)
        assert np.array_equal(np.asarray(source.meta['val_mask'], dtype=bool), val_mask)
        assert kwargs.get('seed', 42) == split['seed'] == 42
        super().__init__(dataset_path=dataset_path, normalizer_mode='legacy', **kwargs)
        self.train_mask, self.val_mask = train_mask, val_mask
        self.sampler = self._make_sampler(train_mask)
        self.normalizer_mode = 'train_only_joint'

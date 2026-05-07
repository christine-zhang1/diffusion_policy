from typing import Dict, Optional, Sequence
import copy
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class SimtoolImageDataset(BaseImageDataset):
    """Stage5 simtool zarr -> diffusion-policy image dataset.

    state layout (140-D), see observation_action_utils_sharpa.OBS_NAME_TO_NAMES:
      [0:29]    joint_pos
      [29:58]   joint_vel
      [58:87]   prev_action_targets
      [87:90]   palm_pos
      [90:94]   palm_rot
      [94:98]   object_rot
      [98:110]  keypoints_rel_palm
      [110:122] keypoints_rel_goal
      [122:137] fingertip_pos_rel_palm
      [137:140] object_scales

    state_dim selects a prefix of that vector. 29 = joint_pos only,
    58 = joint_pos + joint_vel. Anything beyond 87 starts including
    privileged sim state and defeats the point of the visual encoder.
    """

    def __init__(self,
            zarr_path: str,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            state_dim: int = 29,
            image_shape: Sequence[int] = (3, 96, 96),
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])

        full_state_dim = self.replay_buffer['state'].shape[1]
        assert 1 <= int(state_dim) <= full_state_dim, (
            f"state_dim={state_dim} not in [1, {full_state_dim}]")
        image_shape = tuple(int(x) for x in image_shape)
        assert len(image_shape) == 3 and image_shape[0] == 3, (
            f"image_shape must be (3, H, W), got {image_shape}")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.state_dim = int(state_dim)
        self.image_shape = image_shape

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :self.state_dim],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, :self.state_dim].astype(np.float32)
        # (T, H, W, 3) uint8 -> (T, 3, H, W) float32 in [0, 1]
        image = np.moveaxis(sample['img'], -1, 1).astype(np.float32) / 255.0
        _, target_h, target_w = self.image_shape
        if image.shape[-2:] != (target_h, target_w):
            image_t = torch.from_numpy(image)
            image_t = F.interpolate(
                image_t, size=(target_h, target_w),
                mode='bilinear', align_corners=False)
            image = image_t.numpy()
        return {
            'obs': {
                'image': image,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

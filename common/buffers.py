import gym
import torch
import numpy as np
from typing import Sequence, Type, Optional, List, Union


class ReplayBuffer:
    def __init__(self, obs_dim: Union[int, Sequence[int]], act_dim: int, capacity: int, batch_size: int):

        # Transfer the "int" observation dimension to "list"
        if isinstance(obs_dim, int):
            self.obs_dim = [obs_dim]
        else:
            self.obs_dim = list(obs_dim)

        self.act_dim = act_dim
        self.max_size = capacity
        self.batch_size = batch_size
        self.ptr = 0  # Point to the current position in the buffer
        self.crt_size = 0  # The current size of the buffer

        # Use numpy.ndarray to initialize the replay buffer
        self.obs = np.zeros(shape=[self.max_size] + self.obs_dim, dtype=np.float32)
        self.acts = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.rews = np.zeros(self.max_size, dtype=np.float32)
        self.next_obs = np.zeros(shape=[self.max_size] + self.obs_dim, dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.choice(self.crt_size, size=self.batch_size, replace=False)
        return dict(obs=torch.FloatTensor(self.obs[ind]),
                    acts=torch.FloatTensor(self.acts[ind]),
                    rews=torch.FloatTensor(self.rews[ind]),  # 1D
                    next_obs=torch.FloatTensor(self.next_obs[ind]),
                    done=torch.FloatTensor(self.done[ind]))  # 1D


class OfflineBuffer:
    """
    Used in offline setting
    """
    def __init__(self, data: dict, batch_size: int):
        self.obs = data["obs"]
        self.acts = data["acts"]
        self.rews = data["rews"]
        self.next_obs = data["next_obs"]
        self.done = data["done"]

        self.data_num = self.acts.shape[0]
        self.batch_size = batch_size

    def sample(self) -> dict:
        ind = np.random.choice(self.data_num, size=self.batch_size, replace=False)
        return dict(obs=torch.FloatTensor(self.obs[ind]),
                    acts=torch.FloatTensor(self.acts[ind]),
                    rews=torch.FloatTensor(self.rews[ind]),  # 1D
                    next_obs=torch.FloatTensor(self.next_obs[ind]),
                    done=torch.FloatTensor(self.done[ind]))  # 1D


class OfflineBufferAtari:
    """
    Used in offline setting
    """
    def __init__(self, data: dict, batch_size: int):
        self.obs = data["obs"]  # list
        self.acts = data["acts"]  # ndarray
        self.rews = data["rews"]  # ndarray
        self.done = data["done"]  # ndarray

        self.data_num = self.acts.shape[0]
        self.batch_size = batch_size

    def sample(self) -> dict:
        ind = np.random.choice(self.data_num-1, size=self.batch_size, replace=False)
        obs = [self.obs[i] for i in ind]
        next_obs = [self.obs[i+1] for i in ind]
        return dict(obs=torch.FloatTensor(obs),
                    acts=torch.FloatTensor(self.acts[ind]).reshape(-1, 1),
                    rews=torch.FloatTensor(self.rews[ind]),  # 1D
                    next_obs=torch.FloatTensor(next_obs),
                    done=torch.FloatTensor(self.done[ind]))  # 1D

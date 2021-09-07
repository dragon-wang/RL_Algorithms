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
        self.obs = np.zeros(shape=[self.max_size] + self.obs_dim)
        self.acts = np.zeros((self.max_size, self.act_dim))
        self.rews = np.zeros(self.max_size)
        self.next_obs = np.zeros(shape=[self.max_size] + self.obs_dim)
        self.done = np.zeros(self.max_size)

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

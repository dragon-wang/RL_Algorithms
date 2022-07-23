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
        ind = np.random.choice(self.crt_size, size=self.batch_size, replace=True)  # replace=False will make sample be slow
        return dict(obs=torch.FloatTensor(self.obs[ind]),
                    acts=torch.FloatTensor(self.acts[ind]),
                    rews=torch.FloatTensor(self.rews[ind]),  # 1D
                    next_obs=torch.FloatTensor(self.next_obs[ind]),
                    done=torch.FloatTensor(self.done[ind]))  # 1D


class TrajectoryBuffer:
    """
    Used to store experiences for a trajectory (e.g., in PPO)
    """
    def __init__(self, obs_dim: Union[int, Sequence[int]], act_dim: int, capacity: int):

        # Transfer the "int" observation dimension to "list"
        if isinstance(obs_dim, int):
            self.obs_dim = [obs_dim]
        else:
            self.obs_dim = list(obs_dim)
        self.act_dim = act_dim
        self.max_size = capacity
        self.ptr = 0  # Point to the current position in the buffer

        # Use numpy.ndarray to initialize the replay buffer
        self.obs = np.zeros(shape=[self.max_size] + self.obs_dim, dtype=np.float32)
        self.acts = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.rews = np.zeros(self.max_size, dtype=np.float32)
        self.done = np.zeros(self.max_size, dtype=np.float32)
        self.log_probs = np.zeros(self.max_size, dtype=np.float32)  # the log probability of choosing an action
        self.values = np.zeros(self.max_size + 1, dtype=np.float32)  # the value of the state. the length of values is T+1, while others are T
        self.rets = np.zeros(self.max_size, dtype=np.float32)  # the Return in time t, which is also known as G_t.
        self.gae_advs = np.zeros(self.max_size, dtype=np.float32)  # the GAE advantage

    def add(self, obs, act, rew, done, log_prob, value):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.done[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def finish_path(self, last_val=0, gamma=0.99, gae_lambda=0.95, gae_normalize=False):
        """
        This method is called at the end of a trajectory
        """
        self.values[-1] = last_val

        g = self.values[-1]
        gae_adv = 0
        for i in reversed(range(len(self.rews))):
            # compute G_t
            g = self.rews[i] + gamma * g * (1-self.done[i])
            self.rets[i] = g
            # compute A_t
            delt = self.rews[i] + gamma * self.values[i + 1] * (1 - self.done[i]) - self.values[i]
            gae_adv = delt + gamma * gae_lambda * gae_adv * (1 - self.done[i])
            self.gae_advs[i] = gae_adv

        if gae_normalize:
            self.gae_advs = (self.gae_advs - np.mean(self.gae_advs) / np.std(self.gae_advs))

        self.ptr = 0

    def sample(self):
        return dict(obs=torch.FloatTensor(self.obs),
                    acts=torch.FloatTensor(self.acts),
                    rews=torch.FloatTensor(self.rews),
                    done=torch.FloatTensor(self.done),
                    log_probs=torch.FloatTensor(self.log_probs),
                    gae_advs=torch.FloatTensor(self.gae_advs),
                    rets=torch.FloatTensor(self.rets))


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
        ind = np.random.choice(self.data_num, size=self.batch_size, replace=True)  # replace=False will make sample be slow
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
        ind = np.random.choice(self.data_num-1, size=self.batch_size, replace=True)  # replace=False will make sample be slow
        obs = [self.obs[i] for i in ind]
        next_obs = [self.obs[i+1] for i in ind]
        return dict(obs=torch.FloatTensor(obs),
                    acts=torch.FloatTensor(self.acts[ind]).reshape(-1, 1),
                    rews=torch.FloatTensor(self.rews[ind]),  # 1D
                    next_obs=torch.FloatTensor(next_obs),
                    done=torch.FloatTensor(self.done[ind]))  # 1D


class OfflineToOnlineBuffer:
    """
    Used in offline to Online setting
    """
    def __init__(self, data: dict, batch_size: int):
        self.obs = data["obs"]
        self.acts = data["acts"]
        self.rews = data["rews"]
        self.next_obs = data["next_obs"]
        self.done = data["done"]

        self.data_num = self.acts.shape[0]
        self.batch_size = batch_size
        self.max_size = self.data_num

        self.ptr = 0  # Point to the current position in the buffer
        self.crt_size = 0  # The current size of the buffer

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self) -> dict:
        ind = np.random.choice(self.data_num, size=self.batch_size, replace=True)  # replace=False will make sample be slow
        return dict(obs=torch.FloatTensor(self.obs[ind]),
                    acts=torch.FloatTensor(self.acts[ind]),
                    rews=torch.FloatTensor(self.rews[ind]),  # 1D
                    next_obs=torch.FloatTensor(self.next_obs[ind]),
                    done=torch.FloatTensor(self.done[ind]))  # 1D

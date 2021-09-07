import gym
from gym import Env
from tqdm import tqdm
from common.buffers import ReplayBuffer
import numpy as np


def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())


def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


def explore_before_train(env:Env, buffer, explore_step):
    obs = env.reset()
    done = False
    t = tqdm(range(explore_step))
    t.set_description("explore before train")
    for _ in t:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)

        if done:
            obs = env.reset()
            done = False
        else:
            obs = next_obs


class OrnsteinUhlenbeckActionNoise:
    """
    used in DDPG. OU noise
    """
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


if __name__ == '__main__':
    a = OrnsteinUhlenbeckActionNoise(action_dim=3)
    print(a.sample())

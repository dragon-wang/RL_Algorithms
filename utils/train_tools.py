import gym
from gym import Env
from tqdm import tqdm
from common.buffers import ReplayBuffer
import numpy as np
import copy

EVAL_SEED = 10  # used for evaluation env's seed


def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())


def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


def explore_before_train(env: Env, buffer, explore_step):
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


def evaluate(agent, episode_num, seed_offset=100, show=False):
    if show:
        agent.load_agent_checkpoint()
    eval_env = copy.deepcopy(agent.env)
    eval_env.seed(EVAL_SEED + seed_offset)  # reset environment's seed for evaluate(the seed will not be copied by deepcopy)
    total_reward = 0
    total_length = 0
    print("---------------------------------- evaluating at time step {} ----------------------------------".format(agent.train_step))
    for i in range(episode_num):
        episode_reward = 0
        episode_length = 0
        obs, done = eval_env.reset(), False
        while not done:
            if show:
                eval_env.render()
            action = agent.choose_action(obs, eval=True)
            action = action[0] if isinstance(action, tuple) else action
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            if done:
                total_reward += episode_reward
                total_length += episode_length
                if show:
                    print("episode:{} \t step length: {} \t reward: {:.2f}".format(i + 1, episode_length, episode_reward))

    avg_reward = total_reward / episode_num
    avg_length = total_length / episode_num

    print("=====> evaluate {} episode <===> average step length: {:.2f} <===> average reward: {:.2f} <=====".format(episode_num, avg_length, avg_reward))
    print("---------------------------------------------------------------------------------------------------")

    return avg_reward, avg_length


def evaluate_unity(agent, episode_num):
    agent.load_agent_checkpoint()
    eval_env = agent.env
    total_reward = 0
    total_length = 0
    print("---------------------------------- evaluating at time step {} ----------------------------------".format(agent.train_step))
    for i in range(episode_num):
        episode_reward = 0
        episode_length = 0
        obs, done = eval_env.reset(), False
        while not done:
            action = agent.choose_action(obs, eval=True)
            action = action[0] if isinstance(action, tuple) else action
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            if done:
                total_reward += episode_reward
                total_length += episode_length
                print("episode:{} \t step length: {} \t reward: {:.2f}".format(i + 1, episode_length, episode_reward))

    avg_reward = total_reward / episode_num
    avg_length = total_length / episode_num

    print("=====> evaluate {} episode <===> average step length: {:.2f} <===> average reward: {:.2f} <=====".format(episode_num, avg_length, avg_reward))
    print("---------------------------------------------------------------------------------------------------")

    return avg_reward, avg_length


class OrnsteinUhlenbeckActionNoise:
    """
    used in DDPG. OU noise
    """
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
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

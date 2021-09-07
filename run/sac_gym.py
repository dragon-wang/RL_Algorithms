import sys
sys.path.append("..")

import gym
import torch
import torch.nn as nn
from algos.sac import SAC_Agent
from common.buffers import ReplayBuffer
from common.networks import MLPQsaNet, MLPSquashedReparamGaussianPolicy


if __name__ == '__main__':
    # create environment
    env = gym.make("Pendulum-v0")
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('BipedalWalker-v3')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    # create nets
    policy_net = MLPSquashedReparamGaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                                                  hidden_size=[256, 256], hidden_activation=nn.ReLU)
    q_net1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256],
                       hidden_activation=nn.ReLU)

    q_net2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256],
                       hidden_activation=nn.ReLU)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=4e-3)
    q_optimizer1 = torch.optim.Adam(q_net1.parameters(), lr=4e-3)
    q_optimizer2 = torch.optim.Adam(q_net2.parameters(), lr=4e-3)

    # create buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 capacity=50000,
                                 batch_size=128)

    agent = SAC_Agent(env,
                      replay_buffer=replay_buffer,
                      policy_net = policy_net,
                      q_net1 = q_net1,  # critic
                      q_net2 = q_net2,
                      policy_optimizer = policy_optimizer,
                      q_optimizer1 = q_optimizer1,
                      q_optimizer2 = q_optimizer2,
                      gamma=0.99,
                      tau=0.05,
                      alpha=0.5,
                      automatic_entropy_tuning=False,
                      explore_step=2000,
                      max_train_step=50000,
                      train_id="sac_test",
                      log_interval=1000,
                      resume=False)

    agent.learn()
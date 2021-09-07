import sys
sys.path.append("..")

import gym
import torch
import torch.nn as nn
from algos.dqn import DQN_Agent
from algos.ddpg import DDPG_Agent
from common.buffers import ReplayBuffer
from common.networks import MLPQsaNet, DDPGMLPActor

if __name__ == '__main__':
    # create environment
    env = gym.make("Pendulum-v0")
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('BipedalWalker-v3')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    # create nets
    actor_net = DDPGMLPActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                             hidden_size=[400, 300], hidden_activation=nn.ReLU)

    critic_net = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim,
                           hidden_size=[400, 300], hidden_activation=nn.ReLU)

    # create optimizer
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=1e-3)

    # create buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 capacity=50000,
                                 batch_size=64)

    # create agent
    agent = DDPG_Agent(env=env, replay_buffer=replay_buffer,
                       actor_net=actor_net, critic_net=critic_net,
                       actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                       gamma=0.99,
                       tau=0.005,  # used to update target network, w' = tau*w + (1-tau)*w'
                       gaussian_noise_sigma=0.1,
                       explore_step=1000,
                       max_train_step=100000,
                       train_id="ddpg_BipedalWalker_test",
                       log_interval=1000,
                       resume=False
                       )

    agent.learn()

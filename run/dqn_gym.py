import sys
sys.path.append("..")

import gym
import torch
import torch.nn as nn
from algos.dqn import DQN_Agent
from common.buffers import ReplayBuffer
from common.networks import MLPQsNet


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    Q_net = MLPQsNet(obs_dim=obs_dim, act_dim=act_dim,
                     hidden_size=[256, 256], hidden_activation=nn.ReLU)

    optimizer = torch.optim.Adam(Q_net.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1,
                                 capacity=5000, batch_size=32)

    agent = DQN_Agent(env=env,
                      replay_buffer=replay_buffer,
                      Q_net=Q_net,
                      optimizer=optimizer,
                      gamma=0.99,
                      initial_eps=0.1,
                      end_eps=0.001,
                      eps_decay_period=2000,
                      eval_eps=0.001,
                      target_update_freq=10,
                      train_interval=1,
                      explore_step=500,
                      max_train_step=10000,
                      train_id="dqn_CartPole_test",
                      log_interval=500,
                      resume=False,
                      )

    agent.learn()


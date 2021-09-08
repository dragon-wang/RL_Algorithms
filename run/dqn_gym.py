import sys
sys.path.append("..")

import argparse
import gym
import torch
import torch.nn as nn
from algos.dqn import DQN_Agent
from common.buffers import ReplayBuffer
from common.networks import MLPQsNet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN algorithm in gym environment')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='the name of environment')
    parser.add_argument('--capacity', type=int, default=5000,
                        help='the max size of data buffer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the size of batch that sampled from buffer')
    parser.add_argument('--explore_step', type=int, default=500,
                        help='the steps of exploration before train')
    parser.add_argument('--max_train_step', type=int, default=10000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=500,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether load the last saved model to train')
    parser.add_argument('--train_id', type=str, default='dqn_test',
                        help='Path to save model and log tensorboard')

    args = parser.parse_args()

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    Q_net = MLPQsNet(obs_dim=obs_dim, act_dim=act_dim,
                     hidden_size=[256, 256], hidden_activation=nn.ReLU)

    optimizer = torch.optim.Adam(Q_net.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1,
                                 capacity=args.capacity, batch_size=args.batch_size)

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
                      explore_step=args.explore_step,
                      max_train_step=args.max_train_step,
                      train_id=args.train_id,
                      log_interval=args.log_interval,
                      resume=args.resume,
                      )

    agent.learn()


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from algos.ddpg import DDPG_Agent
from common.buffers import ReplayBuffer
from common.networks import MLPQsaNet, DDPGMLPActor
from utils import train_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG algorithm in gym environment')
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='the name of environment')
    parser.add_argument('--capacity', type=int, default=50000,
                        help='the max size of data buffer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='the size of batch that sampled from buffer')
    parser.add_argument('--explore_step', type=int, default=2000,
                        help='the steps of exploration before train')
    parser.add_argument('--max_train_step', type=int, default=100000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='ddpg_gym_test',
                        help='Path to save model and log tensorboard')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether load the last saved model to train')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose cpu or cuda')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the agent')
    parser.add_argument('--seed', type=int, default=10,
                        help='the random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create environment
    env = gym.make(args.env)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    # create nets
    actor_net = DDPGMLPActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                             hidden_size=[400, 300], hidden_activation=nn.ReLU)

    critic_net = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim,
                           hidden_size=[400, 300], hidden_activation=nn.ReLU)

    # create buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 capacity=args.capacity,
                                 batch_size=args.batch_size)

    # create agent
    agent = DDPG_Agent(env=env, replay_buffer=replay_buffer,
                       actor_net=actor_net, critic_net=critic_net,
                       actor_lr=1e-4, critic_lr=1e-3,
                       gamma=0.99,
                       tau=0.005,
                       gaussian_noise_sigma=0.1,
                       explore_step=args.explore_step,
                       max_train_step=args.max_train_step,
                       train_id=args.train_id,
                       log_interval=args.log_interval,
                       resume=args.resume,
                       device=args.device
                       )

    if args.eval:
        train_tools.evaluate(agent, 10, render=True)
    else:
        agent.learn()

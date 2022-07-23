import sys
import os
from pathlib import Path
# sys.path.append(str(Path(__file__).absolute().parent.parent))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import numpy as np
from algos.ppo import PPO_Agent
from common.buffers import TrajectoryBuffer
from common.networks import MLPVsNet, MLPCategoricalActor, MLPGaussianActor
from utils import train_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO algorithm in gym environment')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='the name of environment')
    parser.add_argument('--gae_norm', action='store_true', default=False,
                        help='whether normalize the GAE')
    parser.add_argument('--traj_length', type=int, default=128,
                        help='the length of trajectory')
    parser.add_argument('--eval_freq', type=int, default=1000,
                        help='how often (time steps) we evaluate during training, and it will not eval if eval_freq < 0')
    parser.add_argument('--max_time_step', type=int, default=100000,
                        help='the max time step to train')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='ppo_gym_test',
                        help='Path to save model and log tensorboard')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether load the last saved model to train')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose cpu or cuda')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the trained model visually')
    parser.add_argument('--seed', type=int, default=10,
                        help='the random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create environment
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    train_tools.EVAL_SEED = args.seed

    obs_dim = env.observation_space.shape[0]

    # create nets
    if isinstance(env.action_space, Discrete):
        act_num = env.action_space.n
        buffer_act_dim = 1
        actor_net = MLPCategoricalActor(obs_dim=obs_dim, act_num=act_num,
                                        hidden_size=[64, 64], hidden_activation=nn.Tanh)
    elif isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
        buffer_act_dim = act_dim
        actor_net = MLPGaussianActor(obs_dim=obs_dim, act_dim=act_dim,
                                     hidden_size=[64, 64], hidden_activation=nn.Tanh)

    critic_net = MLPVsNet(obs_dim=obs_dim, hidden_size=[64, 64], hidden_activation=nn.Tanh)


    # create buffer
    if args.show:
        trajectory_buffer = None
    else:
        trajectory_buffer = TrajectoryBuffer(obs_dim=obs_dim,
                                             act_dim=buffer_act_dim,
                                             capacity=args.traj_length)

    agent = PPO_Agent(env,
                      trajectory_buffer=trajectory_buffer,
                      actor_net=actor_net,
                      critic_net=critic_net,
                      actor_lr=3e-4,
                      critic_lr=1e-3,
                      gamma=0.99,
                      gae_lambda=0.95,
                      gae_normalize=args.gae_norm,
                      clip_pram=0.2,
                      trajectory_length=args.traj_length,  # the length of a trajectory_
                      train_actor_iters=10,
                      train_critic_iters=10,
                      eval_freq=args.eval_freq,
                      max_time_step=args.max_time_step,
                      train_id=args.train_id,
                      log_interval=args.log_interval,
                      resume=args.resume,
                      device=args.device)

    if args.show:
        train_tools.evaluate(agent, 10, show=True)
    else:
        agent.learn()

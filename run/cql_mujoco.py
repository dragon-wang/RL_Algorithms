import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from algos.offline.cql import CQL_Agent
from common.buffers import OfflineBuffer
from common.networks import MLPQsaNet, MLPSquashedReparamGaussianPolicy
from utils import train_tools, data_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CQL algorithm in mujoco environment')
    parser.add_argument('--env', type=str, default='hopper-medium-v2',
                        help='the name of environment')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='the size of batch that sampled from buffer')
    parser.add_argument('--auto_alpha_tuning', action='store_true', default=False,
                        help='whether automatic tune alpha')

    # CQL
    parser.add_argument('--min_q_weight', type=float, default=5.0,
                        help='the value of alpha, set to 5.0 or 10.0 if not using lagrange')
    parser.add_argument('--entropy_backup', action='store_true', default=False,
                        help='whether use sac style target Q with entropy')
    parser.add_argument('--max_q_backup', action='store_true', default=False,
                        help='whether use max q backup')
    parser.add_argument('--with_lagrange', action='store_true', default=False,
                        help='whether auto tune alpha in Conservative Q Loss(different from the alpha in sac)')
    parser.add_argument('--lagrange_thresh', type=float, default=10.0,
                        help='the hyper-parameter used in automatic tuning alpha in cql loss')

    parser.add_argument('--max_train_step', type=int, default=2000000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='cql_hopper-medium-v2_test',
                        help='Path to save model and log tensorboard')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether load the last saved model to train')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose cpu or cuda')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the agent')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='the frequency of evaluating the agent in offline')
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
    policy_net = MLPSquashedReparamGaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                                                  hidden_size=[256, 256], hidden_activation=nn.ReLU)
    q_net1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256],
                       hidden_activation=nn.ReLU)

    q_net2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256],
                       hidden_activation=nn.ReLU)

    if args.eval:
        data_buffer = None
    else:
        # create buffer
        data = data_tools.get_d4rl_dataset(env)
        data_buffer = OfflineBuffer(data=data, batch_size=256)

    agent = CQL_Agent(env=env,
                      data_buffer=data_buffer,
                      policy_net=policy_net,
                      q_net1=q_net1,
                      q_net2=q_net2,
                      policy_lr=1e-4,
                      qf_lr=3e-4,
                      gamma=0.99,
                      tau=0.05,
                      alpha=0.5,
                      auto_alpha_tuning=args.auto_alpha_tuning,

                      # CQL
                      min_q_weight=args.min_q_weight,
                      entropy_backup=args.entropy_backup,
                      max_q_backup=args.max_q_backup,
                      with_lagrange=args.with_lagrange,
                      lagrange_thresh=args.lagrange_thresh,
                      n_action_samples=10,

                      max_train_step=args.max_train_step,
                      log_interval=args.log_interval,
                      eval_freq=args.eval_freq,
                      train_id=args.train_id,
                      resume=args.resume,  # if True, train from last checkpoint
                      device=args.device,
                      )
    if args.eval:
        train_tools.evaluate(agent, 10, render=True)
    else:
        agent.learn()
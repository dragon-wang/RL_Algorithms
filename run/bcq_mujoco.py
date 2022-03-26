import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from algos.offline.bcq import BCQ_Agent
from common.buffers import OfflineBuffer
from common.networks import MLPQsaNet, CVAE, BCQ_Perturbation
from utils import train_tools, data_tools


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BCQ algorithm in mujoco environment')
    parser.add_argument('--env', type=str, default='hopper-medium-v2',
                        help='the name of environment')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='the size of batch that sampled from buffer')

    parser.add_argument('--max_train_step', type=int, default=1000000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='bcq_hopper-medium-v2_test',
                        help='Path to save model and log tensorboard')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether load the last saved model to train')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose cpu or cuda')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the trained model visually')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='how often (time steps) we evaluate')
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
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    critic_net1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300],
                            hidden_activation=nn.ReLU)
    critic_net2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[400, 300],
                            hidden_activation=nn.ReLU)

    perturbation_net = BCQ_Perturbation(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                                        hidden_size=[400, 300], hidden_activation=nn.ReLU,
                                        phi=0.05)

    cvae_net = CVAE(obs_dim=obs_dim, act_dim=act_dim,
                    latent_dim=2 * act_dim, act_bound=act_bound)

    # create buffer
    if args.show:
        data_buffer = None
    else:
        data = data_tools.get_d4rl_dataset(env)
        data_buffer = OfflineBuffer(data=data, batch_size=args.batch_size)

    agent = BCQ_Agent(env=env,
                      data_buffer=data_buffer,
                      critic_net1=critic_net1,
                      critic_net2=critic_net2,
                      perturbation_net=perturbation_net,
                      cvae_net=cvae_net,  # generation model
                      critic_lr=1e-3,
                      per_lr=1e-3,
                      cvae_lr=1e-3,

                      gamma=0.99,
                      tau=0.005,
                      lmbda=0.75,  # used for double clipped double q-learning

                      max_train_step=args.max_train_step,
                      log_interval=args.log_interval,
                      eval_freq=args.eval_freq,
                      train_id=args.train_id,
                      resume=args.resume,  # if True, train from last checkpoint
                      device=args.device
                      )

    if args.show:
        train_tools.evaluate(agent, 10, show=True)
    else:
        agent.learn()
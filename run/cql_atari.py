import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import torch
import numpy as np
from algos.offline.cql import DiscreteCQL_Agent
from common.buffers import OfflineBufferAtari
from common.networks import ConvAtariQsNet
from utils import train_tools
from utils import data_tools


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DiscreteCQL algorithm in atari environment')
    parser.add_argument('--env', type=str, default='pong-mixed-v0',
                        help='the name of environment')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='the size of batch that sampled from buffer')

    parser.add_argument('--min_q_weight', type=float, default=5.0,
                        help='the value of alpha, set to 5.0 or 10.0 if not using lagrange')

    parser.add_argument('--max_train_step', type=int, default=2000000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='cql_atari_test',
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

    env = gym.make(args.env, stack=True)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    Q_net = ConvAtariQsNet(num_frames_stack=4, act_dim=act_dim)

    # create buffer
    if args.show:
        data_buffer = None
    else:
        data = data_tools.get_d4rl_dataset_atari(env)
        data_buffer = OfflineBufferAtari(data=data, batch_size=args.batch_size)

    agent = DiscreteCQL_Agent(env=env,
                              data_buffer=data_buffer,
                              Q_net=Q_net,
                              qf_lr=1e-4,
                              gamma=0.99,
                              eval_eps=0.001,
                              target_update_freq=8000,
                              train_interval=1,

                              # CQL
                              min_q_weight=args.min_q_weight,  # the value of alpha in CQL loss, set to 5.0 or 10.0 if not using lagrange

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


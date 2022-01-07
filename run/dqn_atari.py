import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from algos.dqn import DQN_Agent
from common.buffers import ReplayBuffer
from common.networks import ConvAtariQsNet
from utils import train_tools
from utils.atari_wrappers import make_atari_env


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN algorithm in atari environment')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='the name of environment')
    parser.add_argument('--capacity', type=int, default=100000,
                        help='the max size of data buffer')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='the size of batch that sampled from buffer')
    parser.add_argument('--explore_step', type=int, default=20000,
                        help='the steps of exploration before train')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='how often (time steps) we evaluate during train, and it will not eval if eval_freq < 0')
    parser.add_argument('--max_train_step', type=int, default=2000000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether load the last saved model to train')
    parser.add_argument('--train_id', type=str, default='dqn_atari_test',
                        help='Path to save model and log tensorboard')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose cpu or cuda')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the trained model visually')
    parser.add_argument('--seed', type=int, default=10,
                        help='the random seed')
    parser.add_argument('--scale_obs', action='store_true', default=False,
                        help='whether scale the obs to 0-1')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_atari_env(args.env, scale_obs=args.scale_obs)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    Q_net = ConvAtariQsNet(num_frames_stack=4, act_dim=act_dim)

    # create buffer
    if args.show:
        replay_buffer = None
    else:
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1,
                                     capacity=args.capacity, batch_size=args.batch_size)

    agent = DQN_Agent(env=env,
                      replay_buffer=replay_buffer,
                      Q_net=Q_net,
                      qf_lr=1e-4,
                      gamma=0.99,
                      initial_eps=0.1,
                      end_eps=0.001,
                      eps_decay_period=1000000,
                      eval_eps=0.001,
                      target_update_freq=1000,
                      train_interval=1,
                      explore_step=args.explore_step,
                      eval_freq=args.eval_freq,
                      max_train_step=args.max_train_step,
                      train_id=args.train_id,
                      log_interval=args.log_interval,
                      resume=args.resume,
                      device=args.device
                      )

    if args.show:
        train_tools.evaluate(agent, 10, show=True)
    else:
        agent.learn()


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from algos.offline.bear import BEAR_Agent
from common.buffers import OfflineBuffer
from common.networks import MLPQsaNet, MLPSquashedReparamGaussianPolicy, CVAE
from utils import train_tools, data_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CQL algorithm in mujoco environment')
    parser.add_argument('--env', type=str, default='hopper-medium-v0',
                        help='the name of environment')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='the size of batch that sampled from buffer')

    # BEAR
    parser.add_argument('--mmd_sigma', type=float, default=20.0,
                        help='the sigma used in mmd kernel')
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='the type of mmd kernel(gaussian or laplacian)')
    parser.add_argument('--lagrange_thresh', type=float, default=0.05,
                        help='the hyper-parameter used in automatic tuning alpha in cql loss')

    parser.add_argument('--max_train_step', type=int, default=1000000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='bear_hopper-mujoco_test',
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

    # create nets
    policy_net = MLPSquashedReparamGaussianPolicy(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                                                  hidden_size=[256, 256], hidden_activation=nn.ReLU)
    q_net1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256],
                       hidden_activation=nn.ReLU)

    q_net2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim, hidden_size=[256, 256],
                       hidden_activation=nn.ReLU)

    cvae_net = CVAE(obs_dim=obs_dim, act_dim=act_dim,
                    latent_dim=2 * act_dim, act_bound=act_bound)

    # create buffer
    if args.show:
        data_buffer = None
    else:
        data = data_tools.get_d4rl_dataset(env)
        data_buffer = OfflineBuffer(data=data, batch_size=args.batch_size)

    agent = BEAR_Agent(
        # parameters of PolicyBase
        env=env,
        gamma=0.99,
        eval_freq=args.eval_freq,
        max_train_step=args.max_train_step,
        train_id=args.train_id,
        log_interval=args.log_interval,
        resume=args.resume,
        device=args.device,

        # Parameters of OfflineBase
        data_buffer=data_buffer,

        # Parameters of BEAR_Agent
        policy_net=policy_net,
        q_net1=q_net1,
        q_net2=q_net2,
        cvae_net=cvae_net,
        policy_lr=1e-4,
        qf_lr=3e-4,
        cvae_lr=3e-4,
        tau=0.05,
        lmbda=0.75,
        mmd_sigma=args.mmd_sigma,
        kernel_type=args.kernel_type,
        lagrange_thresh=args.lagrange_thresh,
        n_action_samples=100,
        n_target_samples=10,
        n_mmd_action_samples=4,
        warmup_step=40000,
        )
    if args.show:
        train_tools.evaluate(agent, 10, show=True)
    else:
        agent.learn()

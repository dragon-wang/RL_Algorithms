import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from algos.td3 import TD3_Agent
from common.buffers import ReplayBuffer
from common.networks import MLPQsaNet, DDPGMLPActor
from utils import train_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD3 algorithm in mujoco environment')
    parser.add_argument('--env', type=str, default='Hopper-v2',
                        help='the name of environment')
    parser.add_argument('--capacity', type=int, default=1000000,
                        help='the max size of data buffer')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='the size of batch that sampled from buffer')
    parser.add_argument('--explore_step', type=int, default=10000,
                        help='the steps of exploration before train')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='how often (time steps) we evaluate during training, and it will not eval if eval_freq < 0')
    parser.add_argument('--max_train_step', type=int, default=1000000,
                        help='the max train step')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='The number of steps taken to record the model and the tensorboard')
    parser.add_argument('--train_id', type=str, default='td3_mujoco_test',
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
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    # create nets
    actor_net = DDPGMLPActor(obs_dim=obs_dim, act_dim=act_dim, act_bound=act_bound,
                             hidden_size=[400, 300], hidden_activation=nn.ReLU)

    critic_net1 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim,
                            hidden_size=[400, 300], hidden_activation=nn.ReLU)
    critic_net2 = MLPQsaNet(obs_dim=obs_dim, act_dim=act_dim,
                            hidden_size=[400, 300], hidden_activation=nn.ReLU)

    # create buffer
    if args.show:
        replay_buffer = None
    else:
        replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                     act_dim=act_dim,
                                     capacity=args.capacity,
                                     batch_size=args.batch_size)

    agent = TD3_Agent(
        # parameters of PolicyBase
        env=env,
        gamma=0.99,
        eval_freq=args.eval_freq,
        max_train_step=args.max_train_step,
        train_id=args.train_id,
        log_interval=args.log_interval,
        resume=args.resume,
        device=args.device,

        # Parameters of OffPolicyBase
        replay_buffer=replay_buffer,
        explore_step=args.explore_step,

        # Parameters of TD3_Agent
        actor_net=actor_net, critic_net1=critic_net1, critic_net2=critic_net2,
        actor_lr=1e-3, critic_lr=1e-3,  # Or 3e-4
        tau=0.005,
        act_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        )

    if args.show:
        train_tools.evaluate(agent, 10, show=True)
    else:
        agent.learn()

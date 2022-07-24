import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from common.buffers import OfflineBuffer
from utils.train_tools import soft_target_update, evaluate
from utils import log_tools


class TD3_BC_Agent:
    """
    Implementation of TD3 with behavior cloning (TD3_BC)
    https://arxiv.org/abs/2106.06860
    """
    def __init__(self,
                 env,
                 replay_buffer: OfflineBuffer,
                 actor_net: torch.nn.Module,
                 critic_net1: torch.nn.Module,
                 critic_net2: torch.nn.Module,
                 actor_lr=3e-4,
                 critic_lr=3e-4,

                 gamma=0.99,
                 tau=0.005,  # used to update target network, w' = tau*w + (1-tau)*w'
                 policy_noise=0.2,  # Noise added to target policy during critic update
                 noise_clip=0.5,  # Range to clip target policy noise
                 policy_delay=2,  # Frequency of delayed policy updates
                 alpha=2.5,  # The alpha to compute lambda

                 max_train_step=1000000,
                 log_interval=1000,
                 eval_freq=5000,
                 train_id="td3bc_test",
                 resume=False,  # if True, train from last checkpoint
                 device='cpu',
                 ):

        self.env = env
        self.action_num = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.replay_buffer = replay_buffer

        self.device = torch.device(device)

        # the network and optimizers
        self.actor_net = actor_net.to(self.device)
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.critic_net1 = critic_net1.to(self.device)
        self.target_critic_net1 = copy.deepcopy(self.critic_net1).to(self.device)
        self.critic_net2 = critic_net2.to(self.device)
        self.target_critic_net2 = copy.deepcopy(self.critic_net2).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer1 = torch.optim.Adam(self.critic_net1.parameters(), lr=critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.alpha = alpha

        self.actor_loss = 0
        self.eval_freq = eval_freq
        self.max_train_step = max_train_step
        self.train_step = 0

        self.resume = resume  # whether load checkpoint start train from last time

        # log dir and interval
        self.log_interval = log_interval
        self.result_dir = os.path.join(log_tools.ROOT_DIR, "run/results", train_id)
        log_tools.make_dir(self.result_dir)
        self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
        self.tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

    def choose_action(self, obs, eval=False):
        """Choose an action by deterministic policy with some gaussian noise"""
        obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
        with torch.no_grad():
            action = self.actor_net(obs).cpu().numpy().flatten()
        return action

    def train(self):

        # Sample
        batch = self.replay_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Target Policy Smoothing. Add clipped noise to next actions when computing target Q.
        with torch.no_grad():
            noise = torch.normal(mean=0, std=self.policy_noise, size=acts.size()).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_act = self.target_actor_net(next_obs) + noise
            next_act = next_act.clamp(-self.action_bound, self.action_bound)

            # Clipped Double Q-Learning. Compute the min of target Q1 and target Q2
            min_target_q = torch.min(self.target_critic_net1(next_obs, next_act),
                                     self.target_critic_net2(next_obs, next_act)).squeeze(1)
            y = rews + self.gamma * (1. - done) * min_target_q

        current_q1 = self.critic_net1(obs, acts).squeeze(1)
        current_q2 = self.critic_net2(obs, acts).squeeze(1)

        # TD3 Loss
        critic_loss1 = F.mse_loss(current_q1, y)
        critic_loss2 = F.mse_loss(current_q2, y)

        # Optimize critic net
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if (self.train_step+1) % self.policy_delay == 0:
            # Compute actor loss
            pi = self.actor_net(obs)
            Q = self.critic_net1(obs, pi)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, acts)

            # Optimize actor net
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_target_update(self.actor_net, self.target_actor_net, tau=self.tau)
            soft_target_update(self.critic_net1, self.target_critic_net1, tau=self.tau)
            soft_target_update(self.critic_net2, self.target_critic_net2, tau=self.tau)
        else:
            actor_loss = torch.tensor(0)

        self.train_step += 1

        return actor_loss.cpu().item(), critic_loss1.cpu().item(), critic_loss2.cpu().item()

    def learn(self):
        """Train TD3_BC without interacting with the environment (offline)"""
        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)

        while self.train_step < (int(self.max_train_step)):
            actor_loss, critic_loss1, critic_loss2 = self.train()

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                self.tensorboard_writer.log_train_data({"actor_loss": actor_loss,
                                                        "critic_loss1": critic_loss1,
                                                        "critic_loss2": critic_loss2
                                                        }, self.train_step)

            if self.eval_freq > 0 and self.train_step % self.eval_freq == 0:
                avg_reward, avg_length = evaluate(agent=self, episode_num=10)
                self.tensorboard_writer.log_eval_data({"eval_episode_length": avg_length,
                                                       "eval_episode_reward": avg_reward}, self.train_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net1": self.critic_net1.state_dict(),
            "critic_net2": self.critic_net2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer1": self.critic_optimizer1.state_dict(),
            "critic_optimizer2": self.critic_optimizer2.state_dict(),
            "train_step": self.train_step,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.target_actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.target_critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.target_critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])
        self.train_step = checkpoint["train_step"]
        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")
import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
from common.buffers import ReplayBuffer
from gym import Env
from utils.train_tools import soft_target_update, explore_before_train
from utils import log_tools


class SAC_Agent:
    """
    Implementation of Soft Actor-Critic (SAC)
    https://arxiv.org/abs/1812.05905(SAC 2019)
    """
    def __init__(self,
                 env,
                 replay_buffer: ReplayBuffer,
                 policy_net: torch.nn.Module,  # actor
                 q_net1: torch.nn.Module,  # critic
                 q_net2: torch.nn.Module,
                 policy_optimizer: torch.optim.Optimizer,
                 q_optimizer1: torch.optim.Optimizer,
                 q_optimizer2: torch.optim.Optimizer,
                 gamma=0.99,
                 tau=0.05,
                 alpha=0.5,
                 auto_alpha_tuning=False,
                 explore_step=2000,
                 max_train_step=50000,
                 train_id="sac_Pendulum_test",
                 log_interval=1000,
                 resume=False,  # if True, train from last checkpoint
                 device='cpu'
                 ):

        self.env = env
        self.replay_buffer = replay_buffer

        self.device = torch.device(device)

        # the network and optimizers
        self.policy_net = policy_net.to(self.device)
        self.q_net1 = q_net1.to(self.device)
        self.q_net2 = q_net2.to(self.device)
        self.target_q_net1 = copy.deepcopy(self.q_net1).to(self.device)
        self.target_q_net2 = copy.deepcopy(self.q_net2).to(self.device)
        self.policy_optimizer = policy_optimizer
        self.q_optimizer1 = q_optimizer1
        self.q_optimizer2 = q_optimizer2

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha_tuning = auto_alpha_tuning

        if self.auto_alpha_tuning:
            self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-3)
            self.alpha = torch.exp(self.log_alpha)

        self.explore_step = explore_step
        self.max_train_step = max_train_step

        self.train_step = 0
        self.episode_num = 0

        self.resume = resume  # whether load checkpoint start train from last time

        # log dir and interval
        self.log_interval = log_interval
        self.result_dir = os.path.join("./results", train_id)
        log_tools.make_dir(self.result_dir)
        self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
        self.tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

    def choose_action(self, obs, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
            action, log_prob = self.policy_net(obs)

        return action.cpu().numpy().flatten(), log_prob

    def train(self):

        # Sample
        batch = self.replay_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # compute policy Loss
        a, log_prob = self.policy_net(obs)
        min_q = torch.min(self.q_net1(obs, a), self.q_net2(obs, a)).squeeze(1)
        policy_loss = (self.alpha * log_prob - min_q).mean()

        # compute Q Loss
        q1 = self.q_net1(obs, acts).squeeze(1)
        q2 = self.q_net2(obs, acts).squeeze(1)
        with torch.no_grad():
            next_a, next_log_prob = self.policy_net(next_obs)
            min_target_next_q = torch.min(self.target_q_net1(next_obs, next_a), self.target_q_net2(next_obs, next_a)).squeeze(1)
            y = rews + self.gamma * (1. - done) * (min_target_next_q - self.alpha * next_log_prob)

        q_loss1 = F.mse_loss(q1, y)
        q_loss2 = F.mse_loss(q2, y)

        # Update policy network parameter
        # policy network's update should be done before updating q network, or there will make some errors
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update q network1 parameter
        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        # Update q network2 parameter
        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        if self.auto_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0)

        self.train_step += 1

        soft_target_update(self.q_net1, self.target_q_net1, tau=self.tau)
        soft_target_update(self.q_net2, self.target_q_net2, tau=self.tau)

        return q_loss1.cpu().item(), q_loss2.cpu().item(), policy_loss.cpu().item(), alpha_loss.cpu().item()

    def learn(self):
        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)
        explore_before_train(self.env, self.replay_buffer, self.explore_step)
        print("==============================start train===================================")
        obs = self.env.reset()
        done = False

        episode_reward = 0
        episode_length = 0

        while self.train_step < self.max_train_step:
            action, _ = self.choose_action(np.array(obs))
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_length += 1

            q_loss1, q_loss2, policy_loss, alpha_loss = self.train()
            if done:
                obs = self.env.reset()
                done = False
                self.episode_num += 1

                print(f"Total T: {self.train_step} Episode Num: {self.episode_num} "
                      f"Episode Length: {episode_length} Episode Reward: {episode_reward:.3f}")
                self.tensorboard_writer.log_learn_data({"episode_length": episode_length,
                                                        "episode_reward": episode_reward}, self.train_step)
                episode_reward = 0
                episode_length = 0

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                self.tensorboard_writer.log_train_data({"q_loss_1": q_loss1,
                                                        "q_loss_2": q_loss2,
                                                        "policy_loss": policy_loss,
                                                        "alpha_loss": alpha_loss}, self.train_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "q_net1": self.q_net1.state_dict(),
            "q_net2": self.q_net2.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "q_optimizer1": self.q_optimizer1.state_dict(),
            "q_optimizer2": self.q_optimizer2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "train_step": self.train_step,
            "episode_num": self.episode_num
        }
        if self.auto_alpha_tuning:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.q_net1.load_state_dict(checkpoint["q_net1"])
        self.q_net2.load_state_dict(checkpoint["q_net2"])
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.q_optimizer1.load_state_dict(checkpoint["q_optimizer1"])
        self.q_optimizer2.load_state_dict(checkpoint["q_optimizer2"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.train_step = checkpoint["train_step"]
        self.episode_num = checkpoint["episode_num"]
        if self.auto_alpha_tuning:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        print("load checkpoint from " + self.checkpoint_path +
              " and start train from " + str(self.train_step+1) + "step")





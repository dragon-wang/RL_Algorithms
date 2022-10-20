import copy
import numpy as np
import torch
import torch.nn.functional as F
from algos.base import OffPolicyBase
from utils.train_tools import soft_target_update


class DDPG_Agent(OffPolicyBase):
    """
    Implementation of Deep Deterministic Policy Gradient (DDPG)
    https://arxiv.org/abs/1509.02971
    """
    def __init__(self,
                 actor_net: torch.nn.Module,
                 critic_net: torch.nn.Module,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 tau=0.005,  # used to update target network, w' = tau*w + (1-tau)*w'
                 gaussian_noise_sigma=0.2, 
                 **kwargs        
                 ):
        super().__init__(**kwargs)

        self.action_num = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        # the network and optimizers
        self.actor_net = actor_net.to(self.device)
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.critic_net = critic_net.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        self.tau = tau
        self.gaussian_noise_sigma = gaussian_noise_sigma

    def choose_action(self, obs, eval=False):
        """Choose an action by deterministic policy with some gaussian noise"""
        obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
        with torch.no_grad():
            action = self.actor_net(obs).cpu().numpy().flatten()
        if eval:
            return action
        else:
            noise = np.random.normal(0, self.gaussian_noise_sigma, size=self.action_num)
            return (action + noise).clip(-self.action_bound, self.action_bound)

    def train(self):

        # Sample
        batch = self.replay_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Compute target Q value
        with torch.no_grad():
            next_act = self.target_actor_net(next_obs)
            next_Q = self.target_critic_net(next_obs, next_act).squeeze(1)
            target_Q = rews + (1. - done) * self.gamma * next_Q

        # Compute current Q
        current_Q = self.critic_net(obs, acts).squeeze(1)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Compute actor loss
        actor_loss = -self.critic_net(obs, self.actor_net(obs)).mean()

        # Optimize actor net
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize critic net
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        soft_target_update(self.actor_net, self.target_actor_net, tau=self.tau)
        soft_target_update(self.critic_net, self.target_critic_net, tau=self.tau)

        self.train_step += 1

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": critic_loss.cpu().item()}
        return train_summaries

    def store_agent_checkpoint(self):
        checkpoint = {
            "actor_net": self.actor_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "train_step": self.train_step,
            "episode_num": self.episode_num
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.target_actor_net.load_state_dict(checkpoint["actor_net"])
        self.critic_net.load_state_dict(checkpoint["critic_net"])
        self.target_critic_net.load_state_dict(checkpoint["critic_net"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.train_step = checkpoint["train_step"]
        self.episode_num = checkpoint["episode_num"]
        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")

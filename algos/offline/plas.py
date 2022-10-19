import copy
import torch
import torch.nn.functional as F
from algos.base import OfflineBase
from common.networks import MLPQsaNet, CVAE, PLAS_Actor
from utils.train_tools import soft_target_update, evaluate
from utils import log_tools


class PLAS_Agent(OfflineBase):
    """
    Implementation of Policy in the Latent Action Space(PLAS) in continuous action space
    https://arxiv.org/abs/2011.07213
    """
    def __init__(self,
                 critic_net1: MLPQsaNet,
                 critic_net2: MLPQsaNet,
                 actor_net: PLAS_Actor,
                 cvae_net: CVAE,  # generation model
                 critic_lr=1e-3,
                 actor_lr=1e-4,
                 cvae_lr=1e-4,
                 tau=0.005,
                 lmbda=0.75,  # used for double clipped double q-learning
                 max_cvae_iterations=500000,  # the num of iterations when training CVAE model
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.critic_net1 = critic_net1.to(self.device)
        self.critic_net2 = critic_net2.to(self.device)
        self.target_critic_net1 = copy.deepcopy(self.critic_net1).to(self.device)
        self.target_critic_net2 = copy.deepcopy(self.critic_net2).to(self.device)
        self.actor_net = actor_net.to(self.device)
        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.cvae_net = cvae_net.to(self.device)
        self.critic_optimizer1 = torch.optim.Adam(self.critic_net1.parameters(), lr=critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_net2.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.cvae_optimizer = torch.optim.Adam(self.cvae_net.parameters(), lr=cvae_lr)

        self.tau = tau
        self.lmbda = lmbda
        self.max_cvae_iterations = max_cvae_iterations
        self.cvae_iterations= 0
        
    def choose_action(self, obs, eval=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
            action = self.actor_net(obs, self.cvae_net.decode)
        return action.cpu().data.numpy().flatten()

    def train_cvae(self):
        """
        Train CVAE one step
        """
        # Sample
        batch = self.data_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)

        recon_action, mu, log_std = self.cvae_net(obs, acts)
        cvae_loss = self.cvae_net.loss_function(recon_action, acts, mu, log_std)

        self.cvae_optimizer.zero_grad()
        cvae_loss.backward()
        self.cvae_optimizer.step()

        self.cvae_iterations += 1

        train_summaries = {"cvae_loss": cvae_loss.cpu().item()}

        return train_summaries

    def train(self):
        # Sample
        batch = self.data_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        """
        Train Critic
        """
        with torch.no_grad():
            decode_action_next = self.target_actor_net(next_obs, self.cvae_net.decode)

            target_q1 = self.target_critic_net1(next_obs, decode_action_next)
            target_q2 = self.target_critic_net2(next_obs, decode_action_next)

            target_q = (self.lmbda * torch.min(target_q1, target_q2) + (1. - self.lmbda) * torch.max(target_q1, target_q2)).squeeze(1)
            target_q = rews + self.gamma * (1. - done) * target_q

        current_q1 = self.critic_net1(obs, acts).squeeze(1)
        current_q2 = self.critic_net2(obs, acts).squeeze(1)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        """
        Train Actor
        """
        decode_action = self.actor_net(obs, self.cvae_net.decode)
        actor_loss = -self.critic_net1(obs, decode_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """
        Update target networks
        """
        soft_target_update(self.critic_net1, self.target_critic_net1, tau=self.tau)
        soft_target_update(self.critic_net2, self.target_critic_net2, tau=self.tau)
        soft_target_update(self.actor_net, self.target_actor_net, tau=self.tau)

        self.train_step += 1

        train_summaries = {"actor_loss": actor_loss.cpu().item(),
                           "critic_loss": critic_loss.cpu().item()}

        return train_summaries

    def learn(self):
        """Train PLAS without interacting with the environment (offline)"""

        log_tools.make_dir(self.result_dir)
        tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)

        # Train CVAE before train agent
        print("==============================Start to train CVAE==============================")

        while self.cvae_iterations < self.max_cvae_iterations:
            train_summaries_cvae = self.train_cvae()
            if self.cvae_iterations % 1000 == 0:
                print("CVAE iteration:", self.cvae_iterations, "\t", "CVAE Loss:", train_summaries_cvae["cvae_loss"])
                tensorboard_writer.log_train_data(train_summaries_cvae, self.cvae_iterations)

        # Train Agent
        print("==============================Start to train Agent==============================")
        while self.train_step < self.max_train_step:
            train_summaries = self.train()

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                tensorboard_writer.log_train_data(train_summaries, self.train_step)

            if self.eval_freq > 0 and self.train_step % self.eval_freq == 0:
                evaluate_summaries = evaluate(agent=self, episode_num=10)
                tensorboard_writer.log_eval_data(evaluate_summaries, self.train_step)

    def store_agent_checkpoint(self):
        checkpoint = {
            "critic_net1": self.critic_net1.state_dict(),
            "critic_net2": self.critic_net2.state_dict(),
            "actor_net": self.actor_net.state_dict(),
            "cvae_net": self.cvae_net.state_dict(),
            "critic_optimizer1": self.critic_optimizer1.state_dict(),
            "critic_optimizer2": self.critic_optimizer2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "cvae_optimizer": self.cvae_optimizer.state_dict(),
            "train_step": self.train_step,
            "cvae_iterations": self.cvae_iterations,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.critic_net1.load_state_dict(checkpoint["critic_net1"])
        self.critic_net2.load_state_dict(checkpoint["critic_net2"])
        self.actor_net.load_state_dict(checkpoint["actor_net"])
        self.cvae_net.load_state_dict(checkpoint["cvae_net"])
        self.critic_optimizer1.load_state_dict(checkpoint["critic_optimizer1"])
        self.critic_optimizer2.load_state_dict(checkpoint["critic_optimizer2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.cvae_optimizer.load_state_dict(checkpoint["cvae_optimizer"])
        self.train_step = checkpoint["train_step"]
        self.cvae_iterations = checkpoint["cvae_iterations"]

        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")

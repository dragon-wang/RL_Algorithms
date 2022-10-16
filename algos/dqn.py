import copy
import numpy as np
import torch
from algos.base import OffPolicyBase
from utils.train_tools import hard_target_update 


class DQN_Agent(OffPolicyBase):
    """
    Implementation of Deep Q-Network (DQN)
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
    """
    def __init__(self,
                 Q_net: torch.nn.Module,
                 qf_lr=0.001,
                 initial_eps=0.1,
                 end_eps=0.001,
                 eps_decay_period=2000,
                 eval_eps=0.001,
                 target_update_freq =10,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.target_update_freq = target_update_freq

        self.Q_net = Q_net.to(self.device)
        self.target_Q_net = copy.deepcopy(self.Q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=qf_lr)

        # Decay for epsilon
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        self.eval_eps = eval_eps

    def choose_action(self, obs, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.train_step + self.initial_eps, self.end_eps)

        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                return int(self.Q_net(obs).argmax(dim=1).cpu())
        else:
            return self.env.action_space.sample()

    def train(self):
        """
        Sample a batch of data from replay buffer and train
        """

        # Sample
        batch = self.replay_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Compute target Q value
        with torch.no_grad():
            target_q = rews + (1. - done) * self.gamma * self.target_Q_net(next_obs).max(dim=1)[0]

        # Compute current Q value
        current_q = self.Q_net(obs).gather(1, acts.long()).squeeze(1)

        # Compute Q loss
        q_loss = 0.5 * (target_q - current_q).pow(2).mean()
        # Q_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the Q network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        self.train_step += 1

        # update target Q
        if self.train_step % self.target_update_freq == 0:
            hard_target_update(self.Q_net, self.target_Q_net)

        train_summaries = {"q_loss": q_loss.cpu().item()}

        return train_summaries

    def store_agent_checkpoint(self):
        checkpoint = {
            "net": self.Q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "episode_num": self.episode_num
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)  # can load gpu's data on cpu machine
        self.Q_net.load_state_dict(checkpoint["net"])
        self.target_Q_net = copy.deepcopy(self.Q_net)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint["train_step"]
        self.episode_num = checkpoint["episode_num"]
        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")
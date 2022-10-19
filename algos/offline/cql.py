import copy
import numpy as np
import torch
import torch.nn.functional as F
from algos.base import OfflineBase
from utils.train_tools import soft_target_update, hard_target_update


class CQL_Agent(OfflineBase):
    """
    Implementation of Conservative Q-Learning for Offline Reinforcement Learning (CQL)
    https://arxiv.org/abs/2006.04779
    This is CQL based on SAC, which is suitable for continuous action space.
    """
    def __init__(self,
                 policy_net: torch.nn.Module,  # actor
                 q_net1: torch.nn.Module,  # critic
                 q_net2: torch.nn.Module,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 tau=0.05,
                 alpha=0.5,
                 auto_alpha_tuning=False,

                 # CQL
                 min_q_weight=5.0,  # the value of alpha in CQL loss, set to 5.0 or 10.0 if not using lagrange
                 entropy_backup=False,  # whether use sac style target Q with entropy
                 max_q_backup=False,  # whether use max q backup
                 with_lagrange=False,  # whether auto tune alpha in Conservative Q Loss(different from the alpha in sac)
                 lagrange_thresh=0.0,  # the hyper-parameter used in automatic tuning alpha in cql loss
                 n_action_samples=10,  # the number of action sampled in importance sampling
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        # the network and optimizers
        self.policy_net = policy_net.to(self.device)
        self.q_net1 = q_net1.to(self.device)
        self.q_net2 = q_net2.to(self.device)
        self.target_q_net1 = copy.deepcopy(self.q_net1).to(self.device)
        self.target_q_net2 = copy.deepcopy(self.q_net2).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=qf_lr)
        self.q_optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=qf_lr)

        self.tau = tau
        self.alpha = alpha
        self.auto_alpha_tuning = auto_alpha_tuning

        if self.auto_alpha_tuning:
            self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=policy_lr)
            self.alpha = torch.exp(self.log_alpha)

        # CQL
        self.min_q_weight = min_q_weight
        self.entropy_backup = entropy_backup
        self.max_q_backup = max_q_backup
        self.with_lagrange = with_lagrange
        self.lagrange_thresh = lagrange_thresh
        self.n_action_samples = n_action_samples

        if self.with_lagrange:
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_prime_optimizer = torch.optim.Adam([self.log_alpha_prime], lr=qf_lr)

    def choose_action(self, obs, eval=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).reshape(1, -1).to(self.device)
            action, log_prob, mu_action = self.policy_net(obs)

            if eval:
                action = mu_action  # if eval, use mu as the action

        return action.cpu().numpy().flatten(), log_prob

    def get_policy_actions(self, obs, n_action_samples):
        """
        get n*m actions from m obs
        :param obs: m obs
        :param n_action_samples: num of n
        """
        obs_temp = torch.repeat_interleave(obs, n_action_samples, dim=0).to(self.device)
        with torch.no_grad():
            actions, log_probs, _ = self.policy_net(obs_temp)
        return actions, log_probs.reshape(obs.shape[0], n_action_samples, 1)

    def get_actions_values(self, obs, actions, n_action_samples, q_net):
        """
        get n*m Q(s,a) from m obs and n*m actions
        :param obs: m obs
        :param actions: n actions
        :param n_action_samples: num of n
        :param q_net:
        """
        obs_temp = torch.repeat_interleave(obs, n_action_samples, dim=0).to(self.device)
        q = q_net(obs_temp, actions)
        q = q.reshape(obs.shape[0], n_action_samples, 1)
        return q

    def train(self):

        # Sample
        batch = self.data_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        """
        SAC Loss
        """
        # compute policy Loss
        a, log_prob, _ = self.policy_net(obs)
        min_q = torch.min(self.q_net1(obs, a), self.q_net2(obs, a)).squeeze(1)
        policy_loss = (self.alpha * log_prob - min_q).mean()

        # compute Q Loss
        q1 = self.q_net1(obs, acts).squeeze(1)
        q2 = self.q_net2(obs, acts).squeeze(1)
        with torch.no_grad():
            if not self.max_q_backup:
                next_a, next_log_prob, _ = self.policy_net(next_obs)
                min_target_next_q = torch.min(self.target_q_net1(next_obs, next_a),
                                              self.target_q_net2(next_obs, next_a)).squeeze(1)
                if self.entropy_backup:
                    # y = rews + self.gamma * (1. - done) * (min_target_next_q - self.alpha * next_log_prob)
                    min_target_next_q = min_target_next_q - self.alpha * next_log_prob
            else:
                """when using max q backup"""
                next_a_temp, _ = self.get_policy_actions(next_obs, n_action_samples=10)
                target_qf1_values = self.get_actions_values(next_obs, next_a_temp, self.n_action_samples, self.q_net1).max(1)[0]
                target_qf2_values = self.get_actions_values(next_obs, next_a_temp, self.n_action_samples, self.q_net2).max(1)[0]
                min_target_next_q = torch.min(target_qf1_values, target_qf2_values).squeeze(1)

            y = rews + self.gamma * (1. - done) * min_target_next_q

        q_loss1 = F.mse_loss(q1, y)
        q_loss2 = F.mse_loss(q2, y)

        """
        CQL Loss
        Total Loss = SAC loss + min_q_weight * CQL loss
        """
        # Use importance sampling to compute log sum exp of Q(s, a), which is shown in paper's Appendix F.
        random_sampled_actions = torch.FloatTensor(obs.shape[0] * self.n_action_samples, acts.shape[-1]).uniform_(-1, 1).to(self.device)
        curr_sampled_actions, curr_log_probs = self.get_policy_actions(obs, self.n_action_samples)
        # This is different from the paper because it samples not only from the current state, but also from the next state
        next_sampled_actions, next_log_probs = self.get_policy_actions(next_obs, self.n_action_samples)
        q1_rand = self.get_actions_values(obs, random_sampled_actions, self.n_action_samples, self.q_net1)
        q2_rand = self.get_actions_values(obs, random_sampled_actions, self.n_action_samples, self.q_net2)
        q1_curr = self.get_actions_values(obs, curr_sampled_actions, self.n_action_samples, self.q_net1)
        q2_curr = self.get_actions_values(obs, curr_sampled_actions, self.n_action_samples, self.q_net2)
        q1_next = self.get_actions_values(obs, next_sampled_actions, self.n_action_samples, self.q_net1)
        q2_next = self.get_actions_values(obs, next_sampled_actions, self.n_action_samples, self.q_net2)

        random_density = np.log(0.5 ** acts.shape[-1])

        cat_q1 = torch.cat([q1_rand - random_density, q1_next - next_log_probs, q1_curr - curr_log_probs], dim=1)
        cat_q2 = torch.cat([q2_rand - random_density, q2_next - next_log_probs, q2_curr - curr_log_probs], dim=1)

        min_qf1_loss = torch.logsumexp(cat_q1, dim=1).mean()
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1).mean()

        min_qf1_loss = self.min_q_weight * (min_qf1_loss - q1.mean())
        min_qf2_loss = self.min_q_weight * (min_qf2_loss - q2.mean())

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1e6)
            # the lagrange_thresh has no effect on the gradient of policy,
            # but it has an effect on the gradient of alpha_prime
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.lagrange_thresh)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.lagrange_thresh)

            alpha_prime_loss = -(min_qf1_loss + min_qf2_loss) * 0.5

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss.backward(retain_graph=True)  # the min_qf_loss will backward again latter, so retain graph.
            self.alpha_prime_optimizer.step()
        else:
            alpha_prime_loss = torch.tensor(0)

        q_loss1 = q_loss1 + min_qf1_loss
        q_loss2 = q_loss2 + min_qf2_loss

        """
        Update networks
        """
        # Update policy network parameter
        # policy network's update should be done before updating q network, or there will make some errors
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update q network1 parameter
        self.q_optimizer1.zero_grad()
        q_loss1.backward(retain_graph=True)
        self.q_optimizer1.step()

        # Update q network2 parameter
        self.q_optimizer2.zero_grad()
        q_loss2.backward(retain_graph=True)
        self.q_optimizer2.step()

        if self.auto_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0)

        soft_target_update(self.q_net1, self.target_q_net1, tau=self.tau)
        soft_target_update(self.q_net2, self.target_q_net2, tau=self.tau)

        self.train_step += 1
        
        train_summaries = {"actor_loss": policy_loss.cpu().item(),
                           "critic_loss1": q_loss1.cpu().item(),
                           "critic_loss2": q_loss2.cpu().item(),
                           "alpha_loss": alpha_loss.cpu().item(),
                           "alpha_prime_loss": alpha_prime_loss.cpu().item()}

        return train_summaries

    def store_agent_checkpoint(self):
        checkpoint = {
            "q_net1": self.q_net1.state_dict(),
            "q_net2": self.q_net2.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "q_optimizer1": self.q_optimizer1.state_dict(),
            "q_optimizer2": self.q_optimizer2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "train_step": self.train_step,
        }
        if self.auto_alpha_tuning:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        if self.with_lagrange:
            checkpoint["log_alpha_prime"] = self.log_alpha_prime
            checkpoint["alpha_prime_optimizer"] = self.alpha_prime_optimizer.state_dict()

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

        if self.auto_alpha_tuning:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        if self.with_lagrange:
            self.log_alpha_prime = checkpoint["log_alpha_prime"]
            self.alpha_prime_optimizer.load_state_dict(checkpoint["alpha_prime_optimizer"])

        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")


class DiscreteCQL_Agent(OfflineBase):
    """
    This is CQL based on Double DQN, which is suitable for discrete action space.
    Note that in the paper, the discrete CQL is based on QRDQN.
    """

    def __init__(self,
                 Q_net: torch.nn.Module,
                 qf_lr=0.001,
                 eval_eps=0.001,
                 target_update_freq=8000,
                 min_q_weight=5.0,  # the value of alpha in CQL loss, set to 5.0 or 10.0 if not using lagrange
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.Q_net = Q_net.to(self.device)
        self.target_Q_net = copy.deepcopy(self.Q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=qf_lr)

        self.eval_eps = eval_eps
        self.target_update_freq = target_update_freq
        self.min_q_weight= min_q_weight

    def choose_action(self, obs, eval=True):
        if np.random.uniform(0, 1) > self.eval_eps:
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
        batch = self.data_buffer.sample()
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        """
        Compute DDQN Loss
        """
        # Compute target Q value (Double DQN)
        with torch.no_grad():
            next_acts = self.Q_net(next_obs).max(dim=1)[1].unsqueeze(1)  # use Q net to get next actions, rather than target Q net
            target_Q = self.target_Q_net(next_obs).gather(1, next_acts.long()).squeeze(1)
            target_Q = rews + (1. - done) * self.gamma * target_Q

        # Compute current Q value
        current_Q = self.Q_net(obs).gather(1, acts.long()).squeeze(1)

        # Compute DDQN Q loss
        q_loss = 0.5 * (target_Q - current_Q).pow(2).mean()
        # q_loss = F.mse_loss(current_Q, target_Q)

        """
        Compute CQL Loss
        Total Loss = SAC loss + min_q_weight * CQL loss
        """
        # Compute log sum exp of learned policy
        current_Qs = self.Q_net(obs)
        logsumexp = torch.logsumexp(current_Qs, dim=1, keepdim=True)

        # Compute Q values under buffer data distribution
        data_Q = self.Q_net(obs).gather(1, acts.long()).squeeze(1)

        min_qf_loss = self.min_q_weight * (logsumexp - data_Q).mean()
        q_loss = min_qf_loss + q_loss

        # Optimize the Q network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # update target Q
        if self.train_step % self.target_update_freq == 0:
            hard_target_update(self.Q_net, self.target_Q_net)

        self.train_step += 1
        
        train_summaries = {"q_loss": q_loss.cpu().item()}

        return train_summaries

    def store_agent_checkpoint(self):
        checkpoint = {
            "net": self.Q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_agent_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)  # can load gpu's data on cpu machine
        self.Q_net.load_state_dict(checkpoint["net"])
        self.target_Q_net = copy.deepcopy(self.Q_net)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint["train_step"]

        print("load checkpoint from \"" + self.checkpoint_path +
              "\" at " + str(self.train_step) + " time step")

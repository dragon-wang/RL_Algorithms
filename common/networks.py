import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Sequence, Type, Optional, List, Union
from torch.distributions.normal import Normal


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_activation=nn.ReLU, output_activation=nn.Identity):
        super(MLP, self).__init__()
        sizes = [input_dim] + hidden_size + [output_dim]
        layers = []
        for j in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[j], sizes[j + 1]), hidden_activation()]
        layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.as_tensor(x, dtype=torch.float32)
        return self.model(x)


class MLPQsNet(nn.Module):
    """
    DQN Q net
    Input s, output all Q(s, a)
    """
    def __init__(self, obs_dim, act_dim, hidden_size, hidden_activation=nn.ReLU):
        super(MLPQsNet, self).__init__()
        self.mlp = MLP(input_dim=obs_dim,
                       output_dim=act_dim,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)

    def forward(self, obs):
        return self.mlp(obs)


class MLPQsaNet(nn.Module):
    """
    DDPG Critic, SAC Q net, BCQ Critic
    Input (s,a), output Q(s,a)
    """
    def __init__(self, obs_dim, act_dim, hidden_size, hidden_activation=nn.ReLU):
        super(MLPQsaNet, self).__init__()
        self.mlp = MLP(input_dim=obs_dim + act_dim,
                       output_dim=1,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)

    def forward(self, obs, act):
        # obs = torch.as_tensor(obs, dtype=torch.float32)
        # act = torch.as_tensor(act, dtype=torch.float32)
        x = torch.cat([obs, act], dim=1)
        q = self.mlp(x)
        return q


class DDPGMLPActor(nn.Module):
    """
    DDPG Actor
    """
    def __init__(self, obs_dim, act_dim, act_bound, hidden_size, hidden_activation=nn.ReLU):
        super(DDPGMLPActor, self).__init__()
        self.mlp = MLP(input_dim=obs_dim, output_dim=act_dim,
                       hidden_size=hidden_size, hidden_activation=hidden_activation)
        self.act_bound = act_bound

    def forward(self, obs):
        a = torch.tanh(self.mlp(obs))
        a = self.act_bound * a
        return a


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLPSquashedReparamGaussianPolicy(nn.Module):
    """
    Policy net.Used in SAC, CQL.
    Input s, output reparameterize, squashed action and log probability of this action
    """
    def __init__(self, obs_dim, act_dim, act_bound, hidden_size, hidden_activation=nn.ReLU, edge=3e-3):

        super(MLPSquashedReparamGaussianPolicy, self).__init__()

        self.mlp = MLP(input_dim=obs_dim, output_dim=hidden_size[-1], hidden_size=hidden_size[:-1],
                       hidden_activation=hidden_activation, output_activation=hidden_activation)
        self.fc_mu = nn.Linear(hidden_size[-1], act_dim)
        self.fc_log_std = nn.Linear(hidden_size[-1], act_dim)

        # self.fc_mu.weight.data.uniform_(-edge, edge)
        # self.fc_log_std.bias.data.uniform_(-edge, edge)

        self.hidden_activation = hidden_activation
        self.act_bound = act_bound

    def forward(self, obs):
        x = self.mlp(obs)

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)

        action = self.act_bound * a
        log_prob = torch.sum(dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6), dim=1)
        mu_action = self.act_bound * mu  # used in evaluation

        return action, log_prob, mu_action


class ConvAtariQsNet(nn.Module):
    def __init__(self, num_frames_stack, act_dim):
        super(ConvAtariQsNet, self).__init__()
        self.c1 = nn.Conv2d(num_frames_stack, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.l1 = nn.Linear(3136, 512)
        self.l2 = nn.Linear(512, act_dim)

    def forward(self, obs):
        q = F.relu(self.c1(obs))
        q = F.relu(self.c2(q))
        q = F.relu(self.c3(q))
        q = F.relu(self.l1(q.reshape(-1, 3136)))
        # q = F.relu(self.l1(q.flatten(1)))
        q = self.l2(q)
        return q


class BCQ_CVAE(nn.Module):
    """
    Conditional Variational Auto-Encoder(CVAE) used in Batch-Constrained deep Q-learning(BCQ)
    ref: https://github.com/sfujim/BCQ/blob/4876f7e5afa9eb2981feec5daf67202514477518/continuous_BCQ/BCQ.py#L57
    """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 latent_dim,
                 act_bound):
        """
        :param obs_dim: The dimension of observation
        :param act_dim: The dimension if action
        :param latent_dim: The dimension of latent in CVAE
        :param act_bound: The maximum value of the action
        """
        super(BCQ_CVAE, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.act_bound = act_bound

        # encoder net
        self.e1 = nn.Linear(obs_dim + act_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.e3_mu = nn.Linear(750, latent_dim)
        self.e3_log_std = nn.Linear(750, latent_dim)

        # decoder net
        self.d1 = nn.Linear(obs_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, act_dim)

    def encode(self, obs, action):
        h1 = F.relu(self.e1(torch.cat([obs, action], dim=1)))
        h2 = F.relu(self.e2(h1))
        mu = self.e3_mu(h2)
        log_std = self.e3_log_std(h2).clamp(-4, 15)  # Clamped for numerical stability
        return mu, log_std

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def decode(self, obs, z=None, z_device=torch.device('cpu')):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).to(z_device).clamp(-0.5, 0.5)
        h4 = F.relu(self.d1(torch.cat([obs, z], dim=1)))
        h5 = F.relu(self.d2(h4))
        recon_action = torch.tanh(self.d3(h5)) * self.act_bound
        return recon_action

    def forward(self, obs, action):
        mu, log_std = self.encode(obs, action)
        z = self.reparametrize(mu, log_std)
        # std = torch.exp(log_std)
        # dist = Normal(mu, std)
        # z = dist.rsample()
        recon_action = self.decode(obs, z)
        return recon_action, mu, log_std

    def loss_function(self, recon, action, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, action, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + 0.5 * kl_loss
        return loss


class BCQ_Perturbation(nn.Module):
    def __init__(self, obs_dim, act_dim, act_bound, hidden_size, hidden_activation=nn.ReLU,
                 phi=0.05  # the Phi in perturbation model:
                 ):
        super(BCQ_Perturbation, self).__init__()

        self.mlp = MLP(input_dim=obs_dim + act_dim,
                       output_dim=act_dim,
                       hidden_size=hidden_size,
                       hidden_activation=hidden_activation)

        self.act_bound = act_bound
        self.phi = phi

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        a = torch.tanh(self.mlp(x))
        a = self.phi * self.act_bound * a
        return (a + action).clamp(-self.act_bound, self.act_bound)

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
    DDPG Critic, SAC Q net
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
    SAC Policy net
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


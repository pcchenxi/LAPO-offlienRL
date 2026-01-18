import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, latent_dim, max_action):
        super(Actor, self).__init__()
        hidden_size = (256, 256, 256)

        self.pi1 = nn.Linear(state_dim, hidden_size[0])
        self.pi2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.pi3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.pi4 = nn.Linear(hidden_size[2], latent_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.pi1(state))
        a = F.relu(self.pi2(a))
        a = F.relu(self.pi3(a))
        a = self.pi4(a)
        a = self.max_action * torch.tanh(a)

        return a

class ActorVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(ActorVAE, self).__init__()
        hidden_size = (256, 256, 256)

        self.e1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.e2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.e3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.mean = nn.Linear(hidden_size[2], latent_dim)
        self.log_var = nn.Linear(hidden_size[2], latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size[0])
        self.d2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.d3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.d4 = nn.Linear(hidden_size[2], action_dim)  

        self.max_action = max_action
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        z = F.relu(self.e3(z))

        mean = self.mean(z)
        log_var = self.log_var(z)
        std = torch.exp(log_var/2)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, log_var

    def decode(self, state, z=None, clip=None):
        # When sampling from the VAE, the latent vector is clipped
        if z is None:
            clip = self.max_action
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = F.relu(self.d3(a))
        a = self.d4(a)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        hidden_size = (256, 256, 256)

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l4 = nn.Linear(hidden_size[2], 1)

        self.l5 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l6 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l7 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l8 = nn.Linear(hidden_size[2], 1)

        self.v1 = nn.Linear(state_dim, hidden_size[0])
        self.v2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.v3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.v4 = nn.Linear(hidden_size[2], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))

        q2 = F.relu(self.l5(torch.cat([state, action], 1)))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = (self.l8(q2))
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))
        return q1

    def v(self, state):
        v = F.relu(self.v1(state))
        v = F.relu(self.v2(v))
        v = F.relu(self.v3(v))
        v = (self.v4(v))
        return v
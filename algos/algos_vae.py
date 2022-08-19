"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, latent_dim, max_action, device):
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

        return u, z, mean, log_var

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
    def __init__(self, state_dim, action_dim, device):
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

class Latent(object):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, min_v, max_v, replay_buffer, 
                 device, discount=0.99, tau=0.005, vae_lr=1e-4, actor_lr=1e-4, critic_lr=5e-4, 
                 max_latent_action=1, expectile=0.8, kl_beta=1.0, 
                 no_noise=True, doubleq_min=0.8):

        self.device = torch.device(device)
        self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actor_vae_target = copy.deepcopy(self.actor_vae)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr)

        self.actor = Actor(state_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, self.device).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.latent_dim = latent_dim
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.tau_vae = tau

        self.expectile = expectile
        self.kl_beta = kl_beta
        self.no_noise = no_noise
        self.doubleq_min = doubleq_min

        self.replay_buffer = replay_buffer
        self.min_v, self.max_v = min_v, max_v 

    def select_action(self, state):
        with torch.no_grad():
            state = self.replay_buffer.normalize_state(state)
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

            latent_a = self.actor(state)
            action = self.actor_vae_target.decode(state, z=latent_a).cpu().data.numpy().flatten()
                            
            action = self.replay_buffer.unnormalize_action(action)
        return action

    def kl_loss(self, mu, log_var):
        KL_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1).view(-1, 1)
        return KL_loss

    def get_target_q(self, state, actor_net, critic_net, use_noise=False):
        latent_action = actor_net(state)
        if use_noise:
            latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.2, 0.2)
        actor_action = self.actor_vae_target.decode(state, z=latent_action)
        
        target_q1, target_q2 = critic_net(state, actor_action)
        target_q = torch.min(target_q1, target_q2)*self.doubleq_min + torch.max(target_q1, target_q2)*(1-self.doubleq_min)

        return target_q

    def train(self, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

            # Critic Training
            with torch.no_grad():
                next_target_v = self.critic.v(next_state)
                target_Q = reward + not_done * self.discount * next_target_v         
                target_v = self.get_target_q(state, self.actor_target, self.critic_target, use_noise=True)

            current_Q1, current_Q2 = self.critic(state, action)
            current_v = self.critic.v(state)

            v_loss = F.mse_loss(current_v, target_v.clamp(self.min_v, self.max_v))
            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            critic_loss = critic_loss_1 + critic_loss_2 + v_loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute adv and weight
            current_v = self.critic.v(state)

            next_q = self.get_target_q(next_state, self.actor_target, self.critic_target)
            q_action = reward + not_done * self.discount * next_q
            adv = (q_action - current_v)
            weights = torch.where(adv > 0, self.expectile, 1-self.expectile)

            # train weighted CVAE
            recons_action, z_sample, mu, log_var = self.actor_vae(state, action)

            recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)
            KL_loss = self.kl_loss(mu, log_var)
            actor_vae_loss = (recon_loss + KL_loss*self.kl_beta)*weights.detach()
            
            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            actor_vae_loss.backward()
            self.actorvae_optimizer.step()

            # train latent policy 
            latent_actor_action = self.actor(state)
            actor_action = self.actor_vae_target.decode(state, z=latent_actor_action)
            q_pi = self.critic.q1(state, actor_action)
        
            actor_loss = -q_pi.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
                target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)

        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))
        torch.save(self.actor_vae_target.state_dict(), '%s/%s_actor_vae_target.pth' % (directory, filename))


    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename)))

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename)))

        self.actor_vae.load_state_dict(torch.load('%s/%s_actor_vae.pth' % (directory, filename)))
        self.actorvae_optimizer.load_state_dict(torch.load('%s/%s_actor_vae_optimizer.pth' % (directory, filename)))
        self.actor_vae_target.load_state_dict(torch.load('%s/%s_actor_vae_target.pth' % (directory, filename)))
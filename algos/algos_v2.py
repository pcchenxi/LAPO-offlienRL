"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from networks.net_v2 import Actor, Critic, ActorVAE
# from networks.vae_film import FiLMVAE, Critic, Actor
from collections import deque

# from networks.actor_critics import Actor, Critic
# from networks.actor_vae import ActorVAE

class Latent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, min_v, max_v, 
                 device, discount=0.99, tau=0.001, vae_lr=1e-4, actor_lr=1e-4, critic_lr=1e-4, 
                 max_latent_action=3, expectile=0.9, kl_beta=0.5, doubleq_min=0.8):
        super(Latent, self).__init__()

        self.device = torch.device(device)
        # self.actor_vae = FiLMVAE(state_dim, action_dim, latent_dim).to(self.device)
        self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        
        self.actor_vae_target = copy.deepcopy(self.actor_vae)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr, weight_decay=1e-5)

        self.actor = Actor(state_dim, latent_dim, max_latent_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)

        self.latent_dim = latent_dim
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.tau_vae = tau

        self.expectile = expectile
        self.kl_beta = kl_beta
        self.initial_kl_beta = kl_beta
        self.doubleq_min = doubleq_min

        self.min_v, self.max_v = min_v, max_v 

        self.que_length = 100
        self.recons_buff = deque(maxlen=self.que_length)
        # self.recons_buff = deque(maxlen=self.que_length)

        self.use_actor = True


    def copy_bn_param(self):
        source_state = self.actor_vae.state_dict()
        for name, module in self.actor_vae_target.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.copy_(source_state[f'{name}.running_mean'])
                module.running_var.copy_(source_state[f'{name}.running_var'])

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

            if self.use_actor:
                latent_a = self.actor(state)
            else:
                latent_a = None
            # print(latent_a)
            action = self.actor_vae.decode(state, z=latent_a)
            q1, q2 = self.critic(state, action)
            v = self.critic.v(state)

            # a = self.actor_vae.decode(state, z=latent_a)
            # a_targ = self.actor_vae_target.decode(state, z=latent_a)
            # print(a-a_targ)
            
        return action.cpu().data.numpy().flatten(), q1.item(), q2.item(), v.item()

    def kl_loss(self, mu, log_var):
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).view(-1, 1)
        return kld_loss

    def get_pi_q(self, state, actor_net, critic_net, gen_net):
        if self.use_actor:
            latent_action = actor_net(state)
            # latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.2, 0.2)
        else:
            latent_action = None
        # latent_action = None

        # actor_action = gen_net.decode(state, z=latent_action)

        # if use_noise:
        # latent_action += (torch.randn_like(latent_action) * 0.1)#.clamp(-0.2, 0.2)
            # actor_action += (torch.randn_like(actor_action) * 0.2)
        # latent_action = None #torch.randn_like(latent_action)

        actor_action = gen_net.decode(state, z=latent_action)

        target_q1, target_q2 = critic_net(state, actor_action)
        # target_q = torch.min(target_q1, target_q2)*self.doubleq_min + torch.max(target_q1, target_q2)*(1-self.doubleq_min)
        target_q = self.get_min_q(target_q1, target_q2)
        return target_q

    def get_min_q(self, q1, q2):
        # q_mean = (q1+q2)/2
        # q_lb = q_mean - torch.abs(q_mean)*0.01
        # q_min = torch.min(q1, q2)
        # q_combined = torch.min(q1, q2)*self.doubleq_min + torch.max(q1, q2)*(1-self.doubleq_min)
        q = torch.min(q1, q2)*self.doubleq_min + torch.max(q1, q2)*(1-self.doubleq_min)

        return q #torch.max(q_lb, q_min)

    def train_step(self, batch, iter_id, init_beta=True):
        # state, action, next_state, reward, not_done, weight, idx = batch
        state = batch['state'].to(self.device).float()
        action = batch['action'].to(self.device).float()
        next_state = batch['next_state'].to(self.device).float()
        reward = batch['reward'].to(self.device).view(-1,1).float()
        not_done = batch['not_done'].to(self.device).view(-1,1)
        weight = batch['weight'].to(self.device).view(-1,1)

        with torch.no_grad():
            next_target_v = self.critic_target.v(next_state)
            target_q = reward + not_done * self.discount * next_target_v#.clamp(-np.inf, self.max_v)

            # target_v_pi = self.get_pi_q(state, self.actor, self.critic, self.actor_vae_target, use_noise=True)
            target_v_pi = self.get_pi_q(state, self.actor, self.critic, self.actor_vae) #.clamp(-np.inf, self.max_v)
            # action_q1, action_q2 = self.critic(state, action)
            # target_v_a = self.get_min_q(action_q1, action_q2) #.clamp(-np.inf, self.max_v) #.clamp(-np.inf, self.max_v)
            # target_v_a = torch.max(action_q1, action_q2)
            # target_v = torch.max(target_v_pi, target_v_a)
            target_v = target_v_pi
            
            # # adv = target_q - target_v
            # # weight_new = torch.where(adv > 0, self.expectile, 1-self.expectile).detach()

            # target_v_a = self.get_min_q(current_q1, current_q2)
            # adv = target_q - current_v
            # weight = torch.where(adv < 0, 1-self.expectile, self.expectile).detach()
            # target_v = target_v_a * weight

        # Critic Training
        current_q1, current_q2 = self.critic(state, action)
        current_v = self.critic.v(state)

        v_loss = F.mse_loss(current_v, target_v.clamp(self.min_v, self.max_v))
        # v_loss = F.mse_loss(current_v, target_v_a, reduction='none')*weight
        # v_loss = v_loss.mean()
        
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        critic_loss = (critic_loss_1 + critic_loss_2 + v_loss)
    
        # print(critic_loss_1.mean().item(), critic_loss_2.mean().item(), v_loss.mean().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5)
        self.critic_optimizer.step()
        
        loss_rc, loss_kl, a_loss, loss_std, loss_mean= None, None, None, None, None

        if iter_id % 1 == 0:
            with torch.no_grad():
                # q1_a, q2_a = self.critic_target(state, action)
                # q_a = self.get_min_q(q1_a, q2_a)
                # next_q = self.get_pi_q(next_state, self.actor_target, self.critic_target, self.actor_vae_target) 
                # q_a = reward + not_done * self.discount * next_q #self.critic_target.v(next_state)
                # q_a = reward + not_done * self.discount * self.critic.v(next_state)

                # # _, latent_action = self.actor(state)
                # latent_action = None
                # actor_action = self.actor_vae.decode(state, z=latent_action)
                # q1_pi, q2_pi = self.critic(state, actor_action)
                # # # q_pi = torch.min(q1_pi, q2_pi)
                # q_pi = self.get_min_q(q1_pi, q2_pi)
                # q_pi = self.critic.v(state)
                q_pi = self.critic_target.v(state)#.clamp(-np.inf, self.max_v)
                # q_pi = self.get_pi_q(state, self.actor, self.critic, self.actor_vae) #.clamp(-np.inf, self.max_v)
            
                # q_pi = target_v
                # q_pi = target_v_pi
                # q_pi = q1_pi
                # # # adv = q_a - (q_pi-torch.abs(q_pi)*0.01)

                # q_a = self.get_min_q(q1_a, q2_a)
                # # # # q_a = torch.min(q1_a, q2_a)*self.doubleq_min + torch.max(q1_a, q2_a)*(1-self.doubleq_min)
                # q_pi = self.get_pi_q(state, self.actor, self.critic, self.actor_vae)    
                # adv = q_a - q_pi
                adv = target_q - q_pi
                # adv_norm = adv/adv.max() * 10 + 0.1
                # print(adv.max(), adv.mean())
                # adv = q_a - (q_pi-torch.abs(q_pi)*0.01)

                # adv = target_q - self.critic.v(state)
                weight = torch.where(adv < 0, 1-self.expectile, self.expectile).detach()
                # weight = torch.where(adv > 0, self.expectile, 1-self.expectile)

            
            # train weighted CVAE
            recons_action, mu, log_var = self.actor_vae(state, action)
            recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)

            kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # shape: [batch_size, latent_dim]
            kld_loss = kl_per_dim.sum(dim=1).view(-1, 1)
        
            actor_vae_loss = (recon_loss + self.kl_beta * kld_loss)*weight.view(-1,1)

            # actor_vae_loss = (recon_loss + KL_loss*self.kl_beta)*weight.view(-1,1)
            # print(recons_loss_ori.shape, recon_loss.shape, kld_loss_freebits.shape, actor_vae_loss.shape)

            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            actor_vae_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor_vae.parameters(), max_norm=1)
            self.actorvae_optimizer.step()
            # print('update vae')

            # Update Target Networks

            loss_rc = (recons_loss_ori*weight.view(-1,1)).mean().item()
            loss_kl = (kld_loss*weight.view(-1,1)).mean().item()
            a_loss = current_v.mean().item()

        if self.use_actor and iter_id % 1 == 0:
            # train latent policy 
            latent_actor_action = self.actor(state)
            # latent_actor_action = latent_actor_action + torch.randn_like(latent_actor_action) * 0.01

            actor_action = self.actor_vae.decode(state, z=latent_actor_action) #.clamp(-3, 3)
            q1_pi, q2_pi = self.critic(state, actor_action)
            # q1_pi = self.critic.q1(state, actor_action)
            
            # best, use actor_vae for policy training and execution, use target for VAE training
            # q_pi = torch.min(q1_pi, q2_pi)*self.doubleq_min + torch.max(q1_pi, q2_pi)*(1-self.doubleq_min) 
            # a = 0.7
            # q_pi = q1_pi
            q_pi = torch.min(q1_pi, q2_pi)
            # q_pi = self.get_min_q(q1_pi, q2_pi)
            # q_pi = (q1_pi + q2_pi)/2

            actor_qloss = -q_pi.mean()
            # actor_reg_loss = torch.mean(latent_actor_action ** 2)
            actor_loss = actor_qloss #+ actor_reg_loss * 0.001

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
            self.actor_optimizer.step()

            a_loss = -actor_loss.item()
        
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
                target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return weight.cpu().numpy(), a_loss, critic_loss.item(), loss_rc, loss_kl
        # assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)
        # return weight.cpu().numpy(), idx.cpu().numpy(), a_loss, critic_loss_1.item(), critic_loss_2.item(), v_loss.item()
    
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
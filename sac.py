# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 12:13:42 2025

@author: Andy_Wong
"""

import torch
import copy
import numpy as np
import random
from torch.distributions import Normal
import os


def create_mlp(input_size, output_size, hidden_sizes = [256]):
    mlp = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]))
    
    for i in range(len(hidden_sizes)-1):
        mlp.append(torch.nn.ReLU())
        mlp.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
    
    mlp.append(torch.nn.ReLU())
    mlp.append(torch.nn.Linear(hidden_sizes[-1], output_size))
    
    return mlp

class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes = [256]):
        super().__init__()
        self.mlp = create_mlp(state_size + action_size, 1, hidden_sizes)
        
    def forward(self, s, a):
        if(len(s.shape) > 2):
            s = torch.squeeze(s)
        x = torch.cat([s, a], dim=-1)
        
        x = self.mlp(x)
        
        return x


class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes = [256], action_scale = 1.0, action_shift = 0.0):
        super().__init__()
        self.mean_mlp = create_mlp(state_size, action_size, hidden_sizes)
        
        self.log_std_mlp = create_mlp(state_size, action_size, hidden_sizes)
        
        self.tanh = torch.nn.Tanh()
        
        self.action_scale = action_scale
        self.action_shift = action_shift
        
        self.LOG_SIG_MIN = -20.0
        self.LOG_SIG_MAX = 2.0
        
        
    def calculate_action_raw(self, mean, std):
        normal = Normal(mean, std)
        action_raw = normal.rsample()
        
        return action_raw
    
    def calculate_log_prob(self, z, mean, std):
        normal = Normal(mean, std)
        log_prob = normal.log_prob(z)
        
        return log_prob
        
    def forward(self, s):
        mean = self.mean_mlp(s)
        log_std = self.log_std_mlp(s)
        log_std = torch.clamp(log_std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        std = log_std.exp()
        
        action_raw = self.calculate_action_raw(mean, std)

        # Squash the action
        tanh_action = self.tanh(action_raw)
        action = self.action_scale * (tanh_action + self.action_shift)

        log_prob = self.calculate_log_prob(action_raw, mean, std)
        
        # Apply Tanh correction (Jacobian determinant)
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluation(self, s):
        mean = self.mean_mlp(s)
        tanh_action = torch.tanh(mean)
        action = self.action_scale * (tanh_action + self.action_shift)
        
        return action
        

class SAC():
    def __init__(self, lra, lrc, 
                 batch_size, state_size, action_size,
                 device, 
                 target_smoothing_coefficient = 0.005,
                 hidden_sizes_actor = [256], hidden_sizes_critic = [256], 
                 action_scale = 1.0, action_shift = 0.0):
        self.device = device
        
        self.discount = 0.99
        self.target_smoothing_coefficient = target_smoothing_coefficient
        self.entropy_temperature = 0.2
        self.log_alpha = torch.tensor(np.log(self.entropy_temperature), requires_grad=True, device=device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.target_entropy = -self.action_size
        
        self.actor = Actor(state_size, action_size, hidden_sizes_actor, action_scale, action_shift).to(device)
        
        self.critic1 = Critic(state_size, action_size, hidden_sizes_critic).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_sizes_critic).to(device)
        
        self.critic1_target = copy.deepcopy(self.critic1).to(device)
        self.critic2_target = copy.deepcopy(self.critic2).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        
        for param in self.critic2_target.parameters():
            param.requires_grad = False
            
        self.critic1_target.eval()
        self.critic2_target.eval()
         
        self.MSELoss = torch.nn.MSELoss()
        
        self.replayBuffer = []
        self.bufferD_maxsize = 1e6
        
        self.batch_size = batch_size
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lrc)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lrc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lra)
        self.temperature_optimizer = torch.optim.Adam([self.log_alpha], lr=lrc)
    
    def QLoss(self, Q_value, r, d, min_Qtarget_value, log_prob):
        y = r + self.discount * (1 - d) * (min_Qtarget_value.detach() - self.entropy_temperature * log_prob.detach())
        loss = self.MSELoss(Q_value, y)
        
        return loss
    
    def PolicyLoss(self, Q_value, log_prob):
        loss = torch.mean(self.entropy_temperature * log_prob - Q_value)
        
        return loss
    
    def update_Qtarget(self):
        for target_param, source_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                self.target_smoothing_coefficient * source_param.data + (1.0 - self.target_smoothing_coefficient) * target_param.data
            )
        for target_param, source_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                self.target_smoothing_coefficient * source_param.data + (1.0 - self.target_smoothing_coefficient) * target_param.data
            )
    
    def TemperatureLoss(self, log_prob):
        loss = torch.mean(-self.log_alpha * (log_prob + self.target_entropy).detach())
        
        return loss
    
    def update_entropy_temperature(self):
        self.entropy_temperature = np.exp(self.log_alpha.item())
        
    def CPU2GPU_batch(self, data_cpu):
        data_np = np.array(data_cpu, dtype=np.float32)
        data_gpu = torch.tensor(data_np, dtype=torch.float32).to(self.device)
        if len(data_gpu.shape) == 1:
            data_gpu = torch.unsqueeze(data_gpu, -1)
        
        return data_gpu
    
    def train(self):
        # Sample a batch
        [state_batch_cpu, 
         action_batch_cpu, 
         reward_batch_cpu, 
         next_state_batch_cpu, 
         done_batch_cpu] = self.sample_batch(self.batch_size)
        
        state_batch_gpu = self.CPU2GPU_batch(state_batch_cpu)
        action_batch_gpu = self.CPU2GPU_batch(action_batch_cpu)
        reward_batch_gpu = self.CPU2GPU_batch(reward_batch_cpu)
        next_state_batch_gpu = self.CPU2GPU_batch(next_state_batch_cpu)
        done_batch_gpu = self.CPU2GPU_batch(done_batch_cpu)
        
        # Update Q
        with torch.no_grad():
            action_next_state, log_prob_next_state = self.actor(next_state_batch_gpu)
            Q1_target_value = self.critic1_target(next_state_batch_gpu, action_next_state)
            Q2_target_value = self.critic2_target(next_state_batch_gpu, action_next_state)
            min_Qtarget_value = torch.minimum(Q1_target_value, Q2_target_value)
        
        Q1_value = self.critic1(state_batch_gpu, action_batch_gpu)
        Q2_value = self.critic2(state_batch_gpu, action_batch_gpu)
        
        Q1_loss = self.QLoss(Q1_value, 
                             reward_batch_gpu, 
                             done_batch_gpu, 
                             min_Qtarget_value, 
                             log_prob_next_state)
        
        Q2_loss = self.QLoss(Q2_value, 
                             reward_batch_gpu, 
                             done_batch_gpu, 
                             min_Qtarget_value, 
                             log_prob_next_state)
        
        self.critic1_optimizer.zero_grad()
        Q1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        Q2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update Policy
        sample_action, sample_log_prob = self.actor(state_batch_gpu)
        Q1_value_sample = self.critic1(state_batch_gpu, sample_action)
        Q2_value_sample = self.critic2(state_batch_gpu, sample_action)
        min_Qtarget_value = torch.minimum(Q1_value_sample, Q2_value_sample)
        
        policy_loss = self.PolicyLoss(min_Qtarget_value, sample_log_prob)
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy temperature
        entropy_temp_loss = self.TemperatureLoss(sample_log_prob)
        
        self.temperature_optimizer.zero_grad()
        entropy_temp_loss.backward()
        self.temperature_optimizer.step()
        
        self.update_entropy_temperature()
        
        # Update Q target
        self.update_Qtarget()
        
        return Q1_loss.item(), Q2_loss.item(), policy_loss.item(), entropy_temp_loss.item()
        
    def store_buffer(self, state_cpu, action_cpu, reward_cpu, state2_cpu, done_cpu):
        self.replayBuffer.append([state_cpu, action_cpu, reward_cpu, state2_cpu, done_cpu])
        if len(self.replayBuffer) > self.bufferD_maxsize:
            self.replayBuffer.pop(0)
    
    def sample_batch(self, batch_size):
        batch = random.sample(self.replayBuffer, batch_size)
    
        # Unpack the transitions into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)
    
        return [states, actions, rewards, next_states, dones]
    
    def save_checkpoint(self, ckpt_path):
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic1_state_dict': self.critic1.state_dict(),
                    'critic2_state_dict': self.critic2.state_dict(),
                    'critic1_target_state_dict': self.critic1_target.state_dict(),
                    'critic2_target_state_dict': self.critic2_target.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
                    'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
                    'logAlpha': self.log_alpha,
                    'logAlpha_optimizer_state_dict': self.temperature_optimizer.state_dict()
                    }, ckpt_path)
    
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
            self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
            self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
            
            self.log_alpha = checkpoint['logAlpha']
            self.temperature_optimizer.load_state_dict(checkpoint['logAlpha_optimizer_state_dict'])
    
            if evaluate:
                self.actor.eval()
                self.critic1.eval()
                self.critic2.eval()
            else:
                self.actor.train()
                self.critic1.train()
                self.critic2.train()
                
            self.critic1_target.eval()
            self.critic2_target.eval()
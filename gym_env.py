# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:26:53 2025

@author: Andy_Wong
"""

import os
from sac import SAC
import random
from random import randrange
import numpy as np

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

env_name = 'Walker2d-v5'
suffix = 'v1'

eval_mode = False
ckpt_dir = './checkpoints/{}_{}'.format(env_name, suffix)
ckpt_path = ckpt_dir + '/{}_{}.pth'.format(env_name, suffix)
log_path = ckpt_dir + '/runs'
os.makedirs(ckpt_dir, exist_ok=True)

def evaluation(Agent, loops, render_mode = False):
    print('Evaluation:')    
    if render_mode:
        eval_env = gym.make(env_name, render_mode="human")
    else:
        eval_env = gym.make(env_name)
    
    eval_total_reward = 0
    for loop in range(loops):
        eval_state_cpu, info = eval_env.reset()
        
        eval_loop_reward = 0.0
        eval_total_steps = 0
        while True:
            eval_total_steps += 1
            eval_state = torch.tensor(eval_state_cpu, dtype=torch.float32).to(device)
            eval_state = torch.unsqueeze(eval_state, 0)
            
            with torch.no_grad():
                eval_action = Agent.actor.evaluation(eval_state)
                eval_action = torch.squeeze(eval_action)
            eval_action_cpu = eval_action.cpu()
            
            eval_action_cpu = np.array(eval_action_cpu)
            
            eval_state2_cpu, eval_reward, eval_done, eval_truncated, eval_info = eval_env.step(eval_action_cpu)
            eval_loop_reward += eval_reward
            
            if eval_done or eval_truncated:
                eval_total_reward += eval_loop_reward
                print('Total reward:', eval_loop_reward, '\n',
                      'Total steps:', eval_total_steps)
                break
            
            eval_state_cpu = eval_state2_cpu
    avg_eval_reward = eval_total_reward / loops
    print('Average reward:', avg_eval_reward)
    
    eval_env.close()
    
    return avg_eval_reward


if __name__ == "__main__":    
    Env = gym.make(env_name)#, render_mode="human")
    
    action_scale = 1.0
    action_shift = 0.0
    
    Agent = SAC(lra=1e-4,
                lrc=1e-4,
                batch_size=256, 
                state_size=Env.observation_space.shape[0], 
                action_size=Env.action_space.shape[0],
                device=device,
                target_smoothing_coefficient = 0.001,
                hidden_sizes_actor=[256],
                hidden_sizes_critic=[256],
                action_scale=action_scale,
                action_shift=action_shift) # Action = action_scale * (action + action_shift), action:(-1, 1)
    
    if eval_mode:
        Agent.load_checkpoint(ckpt_path)
        
        evaluation(Agent, 30, True)
        
    else:
        Agent.critic1.train()
        Agent.critic2.train()
        Agent.actor.train()
        
        writer = SummaryWriter(log_dir=log_path)
        
        num_epochs = 100000
        num_updates = 1
        random_action = 0.99
        best_reward = 0.0
        for epoch in range(num_epochs):
            print('---------------------------------------------\nEpoch', epoch)
            state_cpu, info = Env.reset()
            
            total_reward = 0
            total_steps = 0
            
            avg_Q1_loss = 0
            avg_Q2_loss = 0
            avg_Policy_loss = 0
            avg_Temp_loss = 0
            num_trains = 0
            while True:
                total_steps += 1
                
                # Get state
                state = torch.tensor(state_cpu, dtype=torch.float32).to(device)
                state = torch.unsqueeze(state, 0)
                
                # Sample action
                if np.random.random() < random_action:
                    action_cpu = []
                    for i in range(Env.action_space.shape[0]):
                        random_float = random.uniform(-1.0, 1.0)
                        random_action = action_scale * (random_float + action_shift)
                        action_cpu.append(random_action)
                    if random_action > 0.05:
                        random_action *= random_action
                else:
                    with torch.no_grad():
                        action, _ = Agent.actor(state)
                        action = torch.squeeze(action)
                    action_cpu = action.cpu()
                
                action_cpu = np.array(action_cpu)
                
                # Update state and get reward
                state2_cpu, reward_cpu, done_cpu, truncated, info = Env.step(action_cpu)
                
                # Save buffer
                Agent.store_buffer(state_cpu, action_cpu, reward_cpu, state2_cpu, done_cpu)
                    
                total_reward += reward_cpu
                
                # Reset if done
                if done_cpu or truncated:
                    print('Total reward:', total_reward, '\n',
                          'Total steps:', total_steps)
                    break
                
                # Update current state
                state_cpu = state2_cpu
                
                # If time to update
                if len(Agent.replayBuffer) >= Agent.batch_size:
                    for update in range(num_updates):                
                        Q1_loss, Q2_loss, policy_loss, entropy_temp_loss = Agent.train()
                        
                        # Store loss
                        avg_Q1_loss += Q1_loss
                        avg_Q2_loss += Q2_loss
                        avg_Policy_loss += policy_loss
                        avg_Temp_loss += entropy_temp_loss
                        num_trains += 1
                    
            if num_trains > 0:
                avg_Q1_loss = avg_Q1_loss/num_trains
                avg_Q2_loss = avg_Q2_loss/num_trains
                avg_Policy_loss = avg_Policy_loss/num_trains
                avg_Temp_loss = avg_Temp_loss/num_trains
                
                print('Training: \n',
                      'avg_Q1_loss:', avg_Q1_loss, '\n',
                      'avg_Q2_loss:', avg_Q2_loss, '\n',
                      'avg_Policy_loss:', avg_Policy_loss, '\n'
                      'avg_Temp_loss:', avg_Temp_loss)
                        
                # Log training metrics to TensorBoard
                writer.add_scalar('Train/avg_Q1_loss', avg_Q1_loss, epoch)
                writer.add_scalar('Train/avg_Q2_loss', avg_Q2_loss, epoch)
                writer.add_scalar('Train/avg_Policy_loss', avg_Policy_loss, epoch)
                writer.add_scalar('Train/avg_Temp_loss', avg_Temp_loss, epoch)
                writer.add_scalar('Train/total_reward', total_reward, epoch)
                writer.add_scalar('Train/total_steps', total_steps, epoch)
                
            # Evaluation
            if (epoch+1)%10 == 0 and len(Agent.replayBuffer) >= Agent.batch_size:
                Agent.actor.train()
                
                avg_eval_reward = evaluation(Agent, 10)
                
                Agent.actor.train()
                
                if avg_eval_reward > best_reward:
                    best_reward = avg_eval_reward
                    print('Best reward:', best_reward)
                    print('Saving model')
                    Agent.save_checkpoint(ckpt_path)
                
                # Log training metrics to TensorBoard
                writer.add_scalar('Eval/avg_eval_reward', avg_eval_reward, epoch)
        Env.close()
        
        # Close TensorBoard writer
        writer.close()
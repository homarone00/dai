from collections import namedtuple, deque
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd




class ActorNet(nn.Module):
    def __init__(self,observation_size,action_size):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,action_size),
            nn.Tanh()
        ).cuda()
        self.apply(init_weights_kaiming)

    def forward(self,observation:torch.Tensor):
        action =  self.net(observation.cuda())
        return action
class CriticNet(nn.Module):
    def __init__(self,observation_size,action_size):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size + action_size,512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512,1),
            nn.Tanh()
        ).cuda()
        self.apply(init_weights_kaiming)

    def forward(self,action:torch.Tensor,observation:torch.Tensor):
        q =  self.net(torch.cat((action.cuda(),observation.cuda()),dim=1).cuda())
        return q


class ReplayBuffer:
    def __init__(self,capacity:int,num_agents:int,batch_size:int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_agents = num_agents
        agent_list = [('walker_' + str(i)) for i in range(num_agents)]
        self.memory = {agent: deque(maxlen=capacity) for agent in agent_list}

    def push(self,agent:str,observation, new_observation,action,reward, termination, truncation, info):
        self.memory[agent].append((observation , new_observation,action,reward, termination, truncation, info))

    def sample(self,agent:str)-> dict:
        batch = random.sample(self.memory[agent], self.batch_size)
        observation , new_observation,action,reward, termination, truncation, info = zip(*batch)

        batch_dict_tensor = {
            'observation': torch.tensor(np.array(observation), dtype=torch.float32),
            'new_observation': torch.tensor(np.array(new_observation), dtype=torch.float32),
            'action':torch.tensor(np.array(action), dtype=torch.float32),
            'reward': torch.tensor(np.array(reward), dtype=torch.float32),
            'termination': torch.tensor(np.array(termination), dtype=torch.bool),
            'truncation': torch.tensor(np.array(truncation), dtype=torch.bool),
            'info': info
        }

        return batch_dict_tensor


    def __len__(self):
        return max(len(buffer) for buffer in self.memory.values())


def soft_update(target_net, source_net, tau=0.005):
    '''
    usage:
    soft_update(target_actor, actor, tau=0.005)
    soft_update(target_critic, critic, tau=0.005)
    '''
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)




def train_critic(batch, nets, optimizer, running_loss,schedulers,gamma=0.99):
    observation, new_observation, action, reward, termination, truncation,info = batch.values()

    # Ensure everything is on the same device
    observation = observation.cuda()
    new_observation = new_observation.cuda()
    reward = reward.unsqueeze(1).cuda()
    stop = (termination | truncation).float().unsqueeze(1).cuda()

    # Compute target Q-value: y = r + γ Q_target(s', a')
    for i in range(1):
        with torch.no_grad():
            next_actions = nets['actor_net_target'](new_observation)  # Compute target actions
            target_q = nets['critic_net_target'](next_actions,new_observation)  # Target Q-value
            y = reward + gamma * target_q * (1 - stop)  # Bellman update

            # Get current Q-value
        q_values = nets['critic_net'](action,observation)  # Critic Q-value

        # Compute loss and update critic
        critic_loss = F.mse_loss(q_values, y)
        running_loss[0]=critic_loss.item()
        running_loss[1]+=1
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(nets['critic_net'].parameters(), max_norm=1.0)  # Prevent exploding gradients
        critic_loss.backward()
        optimizer.step()
        schedulers['critic_net'].step()
    #return critic_loss.item()


def train_actor(batch, nets, optimizer,schedulers):
    observation,_, _,_,_, _, _ = batch.values()  # We only need states

    # Compute actor loss
    for i in range(1):
        # Update actor
        action = nets['actor_net'](observation)  # Get actions from policy
        actor_loss = -nets['critic_net'](action,observation).sum()  # Maximize Q-values
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(nets['actor_net'].parameters(), max_norm=1.0)
        actor_loss.backward()
          # Prevent exploding gradients
        optimizer.step()
        schedulers['actor_net'].step()

    #return actor_loss.item()


def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_models_and_optimizers(all_nets, optims, episode, path="C:\\Users\\omarc\\PycharmProjects\\prova_dai\\multiwalker_v1\\models\\"):
    os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist

    for agent, nets in all_nets.items():
        # Save networks
        torch.save(nets['actor_net'].state_dict(), f"{path}{agent}_actor_{episode}.pth")
        torch.save(nets['critic_net'].state_dict(), f"{path}{agent}_critic_{episode}.pth")
        torch.save(nets['actor_net_target'].state_dict(), f"{path}{agent}_actor_target_{episode}.pth")
        torch.save(nets['critic_net_target'].state_dict(), f"{path}{agent}_critic_target_{episode}.pth")

        # Save optimizers
        torch.save(optims[agent]['actor_net'].state_dict(), f"{path}{agent}_actor_optim_{episode}.pth")
        torch.save(optims[agent]['critic_net'].state_dict(), f"{path}{agent}_critic_optim_{episode}.pth")

    print(f"✅ Saved models & optimizers at episode {episode}")

def load_models_and_optimizers(all_nets, optims, episode, path="C:\\Users\\omarc\\PycharmProjects\\prova_dai\\multiwalker_v1\\models\\"):
    for agent, nets in all_nets.items():
        # Load networks
        nets['actor_net'].load_state_dict(torch.load(f"{path}{agent}_actor_{episode}.pth"))
        nets['critic_net'].load_state_dict(torch.load(f"{path}{agent}_critic_{episode}.pth"))
        nets['actor_net_target'].load_state_dict(torch.load(f"{path}{agent}_actor_target_{episode}.pth"))
        nets['critic_net_target'].load_state_dict(torch.load(f"{path}{agent}_critic_target_{episode}.pth"))

        # Load optimizers
        optims[agent]['actor_net'].load_state_dict(torch.load(f"{path}{agent}_actor_optim_{episode}.pth"))
        optims[agent]['critic_net'].load_state_dict(torch.load(f"{path}{agent}_critic_optim_{episode}.pth"))

    print(f"✅ Loaded models & optimizers from episode {episode}")

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.10, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu  # Mean
        self.theta = theta  # Speed of mean reversion
        self.sigma = sigma  # Noise scale
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        """Reset the internal state (useful between episodes)."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        """Generate noise using the Ornstein-Uhlenbeck process."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

from collections import deque
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: remove all the .cuda() and replace them with .to(device)
class ActorNet(nn.Module):
    def __init__(self,observation_size,action_size):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_size,bias=False),
            nn.Tanh() #mandatory since the action space is bounded
        ).to(DEVICE)
        #self.apply(init_weights_kaiming) #better stability

    def forward(self,observation:torch.Tensor):
        action =  self.net(observation.to(DEVICE))
        return action

class CriticNet(nn.Module):
    def __init__(self,observation_size,action_size):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size + action_size,128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128,1),
            nn.Tanh()
        ).to(DEVICE)
        self.apply(init_weights_kaiming) #better stability

    def forward(self,action:torch.Tensor,observation:torch.Tensor):
        q =  self.net(torch.cat((action.to(DEVICE),observation.to(DEVICE)),dim=1).to(DEVICE))
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

        batch_dict_tensor = { #never used as dict...maybe i could just return a tuple...
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

def train_critic(batch, nets, optimizer,schedulers,gamma=0.95):
    observation, new_observation, action, reward, termination, truncation,info = batch.values()

    #everything on .cuda()...so bad...I should use .to(device)
    observation = observation.to(DEVICE)
    new_observation = new_observation.to(DEVICE)
    reward = reward.unsqueeze(1).to(DEVICE)
    stop = (termination | truncation).float().unsqueeze(1).to(DEVICE)

    for i in range(1):
        with torch.no_grad():
            next_actions = nets['actor_net_target'](new_observation)  #target actions
            target_q = nets['critic_net_target'](next_actions,new_observation)  #target q value
            q = reward + gamma * target_q * (1 - stop)  #Bellman update

        #current q value
        q_values = nets['critic_net'](action,observation)

        # Compute loss and update critic
        critic_loss = F.mse_loss(q_values, q)

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(nets['critic_net'].parameters(), max_norm=0.5) #fear of exp. gradients
        critic_loss.backward()
        optimizer.step()
        schedulers['critic_net'].step()



def train_actor(batch, nets, optimizer,schedulers):
    observation,_, _,_,_, _, _ = batch.values()

    # Compute actor loss
    for i in range(1):
        # Update actor
        nets['critic_net'].eval()
        action = nets['actor_net'](observation)  #get action
        actor_loss = -nets['critic_net'](action,observation).mean()
        optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(nets['actor_net'].parameters(), max_norm=0.5) # more fear of exp. gradients
        actor_loss.backward()
        optimizer.step()
        schedulers['actor_net'].step()
        nets['critic_net'].train()

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_models_and_optimizers(all_nets, optims, episode, path):
    #os.makedirs(path, exist_ok=True)  #I prefer to let the user make the directory...it seems safer

    for agent, nets in all_nets.items():
        #Save nets
        torch.save(nets['actor_net'].state_dict(), f"{path}/{agent}_actor_{episode}.pth")
        torch.save(nets['critic_net'].state_dict(), f"{path}/{agent}_critic_{episode}.pth")
        torch.save(nets['actor_net_target'].state_dict(), f"{path}/{agent}_actor_target_{episode}.pth")
        torch.save(nets['critic_net_target'].state_dict(), f"{path}/{agent}_critic_target_{episode}.pth")

        #Save optims
        torch.save(optims[agent]['actor_net'].state_dict(), f"{path}/{agent}_actor_optim_{episode}.pth")
        torch.save(optims[agent]['critic_net'].state_dict(), f"{path}/{agent}_critic_optim_{episode}.pth")

    print(f"Saved models and optimizers at episode {episode}! :-)")

def load_models_and_optimizers(all_nets, optims, episode, path):
    for agent, nets in all_nets.items():
        # Load networks
        nets['actor_net'].load_state_dict(torch.load(f"{path}/{agent}_actor_{episode}.pth",map_location=DEVICE))
        nets['critic_net'].load_state_dict(torch.load(f"{path}/{agent}_critic_{episode}.pth",map_location=DEVICE))
        nets['actor_net_target'].load_state_dict(torch.load(f"{path}/{agent}_actor_target_{episode}.pth",map_location=DEVICE))
        nets['critic_net_target'].load_state_dict(torch.load(f"{path}/{agent}_critic_target_{episode}.pth",map_location=DEVICE))

        # Load optimizers
        optims[agent]['actor_net'].load_state_dict(torch.load(f"{path}/{agent}_actor_optim_{episode}.pth",map_location=DEVICE))
        optims[agent]['critic_net'].load_state_dict(torch.load(f"{path}/{agent}_critic_optim_{episode}.pth",map_location=DEVICE))

    print(f"Loaded models & optimizers from episode {episode}! :-)")

class OUNoise: #i tried both gaussian and ou. literature says that ou is better
    def __init__(self, action_dim, mu=0.0, theta=0.10, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu            #mean
        self.theta = theta
        self.sigma = sigma      #scale
        self.state = np.ones(self.action_dim) * self.mu

    def reset_noise(self):

        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        #copied code...let's hope it works
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

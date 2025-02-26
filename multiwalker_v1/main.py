#IMPORTS
from pettingzoo.sisl import multiwalker_v9
from pettingzoo.utils.env import AECEnv
import numpy as np
import os

#TORCH IMPORTS
import torch
import torch.nn as nn
from torch.optim import Adam

from multiwalker_v1.utils import ActorNet, CriticNet
#MY IMPORTS
from utils import (ActorNet, CriticNet, ReplayBuffer, soft_update, train_actor, train_critic,
                   save_models_and_optimizers, load_models_and_optimizers, OUNoise)

#=======================================================================================================
#HYPERPARAMETERS
NUM_WALKERS = 3
N_EPISODES = 20000
OBSERVATION_SIZE = 31
ACTION_SIZE = 4
START_EPISODE = 799 ############################################
LOAD_CHECKPOINT = (START_EPISODE != 0)
RENDER = True
FORWARD_REWARD = 10.0
TERMINATE_REWARD = 0.0
FALL_REWARD  = -50.0
WARMUP = 5000 #number of episodes only used to fill the ReplayMemory, without updates
BUFFER_CAPACITY = 100_000   #replaybuffer
BATCH_SIZE = 512           #replaybuffer
SAVE_FREQ = 100 #frequency of the weight saves
WEIGHTS_ABS_PATH = '/home/omarc/PycharmProjects/dai/multiwalker_v1/models'

#=======================================================================================================
#prints and checks
print('BE CAREFUL!!! THIS SCRIPT WILL EVENTUALLY ERASE ALL THE CONTENTS OF THE WEIGHTS_ABS_PATH DIRECTORY')
assert os.path.isdir(WEIGHTS_ABS_PATH), 'provide a valid abs path to an empty directory'
print(f'directory: {WEIGHTS_ABS_PATH} <-- will be erased!!!!!!!!!!!!')
is_training = 0
#=======================================================================================================
#STARTING THE ENVIRONMENT
env:AECEnv = multiwalker_v9.env(render_mode = 'human' if RENDER else None,  #TODO: use a parallel env
                         n_walkers=NUM_WALKERS,
                         position_noise=1e-3,
                         angle_noise=1e-3,
                         forward_reward=FORWARD_REWARD,
                         terminate_reward=TERMINATE_REWARD,
                         fall_reward=FALL_REWARD,
                         shared_reward=False, #to make the agents fully independent
                         terminate_on_fall=False, #to avoid unnecessary punishment to agents that didn't do anything wrong
                         remove_on_fall=True,
                         terrain_length=200,
                         max_cycles=1000)
#=======================================================================================================
#AGENTS, NETS, OPTIMIZERS AND NOISE
all_agents = [f'walker_{i}' for i in range(NUM_WALKERS)]

all_nets: dict[str, dict[str, ActorNet | CriticNet]] = {
    agent: {
        'critic_net': CriticNet(OBSERVATION_SIZE, ACTION_SIZE),
        'critic_net_target': CriticNet(OBSERVATION_SIZE, ACTION_SIZE),
        'actor_net': ActorNet(OBSERVATION_SIZE, ACTION_SIZE),
        'actor_net_target': ActorNet(OBSERVATION_SIZE, ACTION_SIZE)
    }
    for agent in all_agents
}

optims = {
    agent: {
        'critic_net': Adam(all_nets[agent]['critic_net'].parameters(), lr=0.001),
        'actor_net': Adam(all_nets[agent]['actor_net'].parameters(), lr=0.001)
    }
    for agent in all_agents
}

schedulers = { #should half the lr after 138629 steps
    agent: {
        'critic_net': torch.optim.lr_scheduler.ExponentialLR(optims[agent]['critic_net'], gamma=0.999995),
        'actor_net': torch.optim.lr_scheduler.ExponentialLR(optims[agent]['actor_net'], gamma=0.999995)
    }
    for agent in all_agents
}
if LOAD_CHECKPOINT:
    load_models_and_optimizers(all_nets,optims,START_EPISODE,WEIGHTS_ABS_PATH)

#Setting all nets into the right mode
for agent in all_agents:
    all_nets[agent]['critic_net'].train()
    all_nets[agent]['critic_net_target'].eval()
    all_nets[agent]['actor_net'].train()
    all_nets[agent]['actor_net_target'].eval()
    soft_update(all_nets[agent]['critic_net_target'],all_nets[agent]['critic_net'],1)
    soft_update(all_nets[agent]['actor_net_target'], all_nets[agent]['actor_net'], 1)

ou_noise = {agent: OUNoise(ACTION_SIZE) for agent in all_agents}
#=======================================================================================================
memory = ReplayBuffer(BUFFER_CAPACITY,NUM_WALKERS,BATCH_SIZE)

cache = {
    agent: {
        'observation': None,
        'action':None,
        'reward':None,
        'termination':None,
        'truncation':None,
        'info':None
    }
    for agent in all_agents
}
#=======================================================================================================

for episode in range(START_EPISODE + 1,N_EPISODES):
    #RESETTING THINGS
    episode_rewards = {agent: 0.0 for agent in all_agents}
    env.reset()
    for agent in all_agents:
        ou_noise[agent].reset_noise()
    #AGENT ITERATION
    for agent in env.agent_iter(): #TODO: MAKE IT PARALLEL!
        observation, reward, termination, truncation, info = env.last()
        reward/=500 #normalization of the reward for stability (could be wrong)
        '''
        During my experiments, I noticed that the reward is never > 500 or < -500. Clipping is performed just in case 
        something goes very wrong.
        '''
        reward = np.clip(reward,-1,1)
        if termination or truncation:
            action = None
        else:
            with torch.no_grad():
                if len(memory) < WARMUP and START_EPISODE == 0:
                    action = env.action_space(agent).sample() #old random policy provided by the example
                else:
                    action = all_nets[agent]['actor_net'](torch.tensor(observation, dtype=torch.float32)).cpu().numpy()
                    action += ou_noise[agent].sample()
                    action = np.clip(action, -1, 1)
        episode_rewards[agent]+=reward
        env.step(action)
        if termination or truncation:
            action = np.zeros(ACTION_SIZE) #zeros action, since i have to feed something to the networks. It doesnt matter
                                           #since termination actions aren't used in the bellman update.
        if observation is None:
            observation = np.zeros(OBSERVATION_SIZE)
        if cache[agent]['observation'] is not None:
            memory.push(
                agent,
                cache[agent]['observation'],
                observation,
                cache[agent]['action'],
                cache[agent]['reward'],
                cache[agent]['termination'],
                cache[agent]['truncation'],
                cache[agent]['info'])
        cache[agent]['observation'] = observation
        cache[agent]['action'] = action
        cache[agent]['reward'] = reward
        cache[agent]['termination'] = termination
        cache[agent]['truncation'] = truncation
        cache[agent]['info'] = info
        batch = None

        #UPDATE PART
        if len(memory) > WARMUP:
            batch = memory.sample(agent)
            if is_training == 0:
                is_training = 1
                print('starting training')

            #CRITIC PART
            train_critic(batch, all_nets[agent], optims[agent]['critic_net'],schedulers[agent])
            soft_update(all_nets[agent]['critic_net_target'], all_nets[agent]['critic_net'])

        if len(memory) > 2 * WARMUP:
            #ACTOR PART
            train_actor(batch, all_nets[agent], optims[agent]['actor_net'],schedulers[agent])
            soft_update(all_nets[agent]['actor_net_target'], all_nets[agent]['actor_net'])


    if (episode + 1)% SAVE_FREQ == 0: #saving the checkpoints
        print(f'saving model:{episode}')
        for file in os.listdir(WEIGHTS_ABS_PATH):
            os.remove(os.path.join(WEIGHTS_ABS_PATH,file))
        save_models_and_optimizers(all_nets, optims, episode,path=WEIGHTS_ABS_PATH)
    print(str(episode) + ' -->' + str(episode_rewards))
    env.close()
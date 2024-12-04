import numpy as np
import matplotlib.pyplot as plt
import mujoco
import gymnasium as gym

import random
import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm
import math
import os
import copy


# for copying deep nets to another variable
from copy import deepcopy

# library for ou noise as implemented with the paper
from ou_noise import ou

# to view model summary
from torchsummary import summary

# queue for replay buffer
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        # initialize parameters
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def insert(self, obs, action, reward, next_obs, done):
        # tuple to represent transition
        trans = (torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.float32),
                 torch.tensor(reward, dtype=torch.float32), torch.tensor(next_obs, dtype=torch.float32), torch.tensor(done, dtype=torch.float32))

        # save transition to buffer
        # use deque because once its full it discards old items
        self.buffer.append(trans)

    def sample_random_minibatch(self, batch_size, device):
        # Random idx to sample from buffer w/o replacement
        batch = random.sample(self.buffer, batch_size)

        # Unpack batch into separate lists of tensors
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        # Convert lists of tensors into single tensors
        obs = torch.stack(obs).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.stack(rewards).to(device)
        next_obs = torch.stack(next_obs).to(device)
        dones = torch.stack(dones).to(device)

        # # convert list to tensor for easy slciing
        # batch = torch.tensor(batch)

        # # slicing to grab elements
        # obs = batch[:,0]
        # actions =  batch[:,1]
        # rewards = batch[:,2]
        # next_obs = batch[:,3]
        # dones = batch[:,4]

        # tuple of tensors
        batch = (obs, actions, rewards, next_obs, dones)

        return batch
    
# Actor AKA: The POLICY
class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dims=(400,300), init_weight = 3e3) -> None:
        super(Actor, self).__init__()
        # In the DDPG paper the parameters for the ACTOR are:
        # - Learning rate: 10^-4
        # - 2 hidden layers
        # - 400 & 300 hidden dims (called units in paper) for first and second hidden layer, respectively
        # - ReLU (rectified nonlinearity) for all hidden layers
        # - output layer uses tanh (returns actions needed for the agent)

        # initializing layer weights
        # - hidden layers weights iniitalized with uniform distribution (-1/sqrt(fan_in), 1/sqrt(fan_in)); fan_in being the input of that particular layer
        # - output layer weights initialized with uniform distribution (-3e-3,3e-3)


        self.init_weight_limit = init_weight

        # hidden layers
        self.hidden1 = nn.Linear(num_states, hidden_dims[0]) # input to hidden
        self.hidden2 = nn.Linear(hidden_dims[0], hidden_dims[1]) # hidden to hidden
        # output layer
        self.output = nn.Linear(hidden_dims[1], num_actions) # hidden to output
        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        # input to first hidden layer w/ relu activation
        x = self.hidden1(x)
        x = self.relu(x)

        # feed into second hidden layer w/ relu activation
        x = self.hidden2(x)
        x = self.relu(x)
        
        # feed through output layer w/ tanh activation
        x = self.output(x)
        y = self.tanh(x)

        return y
    
    def init_weights(self):
        # init hidden with uniform distribution (-1/sqrt(fan_in), 1/sqrt(fan_in)); fan_in being the input of that particular layer
        self.hidden1.weight.data.uniform_(-(1/math.sqrt(self.hidden1.weight.size(1))),(1/math.sqrt(self.hidden1.weight.size(1))))
        self.hidden2.weight.data.uniform_(-(1/math.sqrt(self.hidden2.weight.size(1))),(1/math.sqrt(self.hidden2.weight.size(1))))
        # output layer weights init with uniform distribution (-3e-3,3e-3)
        self.output.weight.data.uniform_(-self.init_weight_limit, self.init_weight_limit)

# Critic AKA: The Q-VALUE FUNCTION
class Critic(nn.Module):
    def __init__(self, num_states, num_actions, output_dim=1, hidden_dims=(400,300), init_weight = 3e3) -> None:
        super(Critic, self).__init__()
        # In the DDPG paper the parameters for the CRITIC are:
        # - Learning rate: 10^-3
        # - 2 hidden layers
        # - 400 & 300 hidden dims (called units in paper) for first and second hidden layer, respectively
        # - ReLU (rectified nonlinearity) for all hidden layers
        # - output layer uses tanh (returns a single q-value for the input state-action pair)
        # - output layer weights initialized with uniform distribution (low=-3e-3,high=3e-3)

        # initializing layer weights
        # - hidden layers weights iniitalized with uniform distribution (-1/sqrt(fan_in), 1/sqrt(fan_in)); fan_in being the input of that particular layer
        # - output layer weights initialized with uniform distribution (-3e-3,3e-3)

        self.init_weight_limit = init_weight

        # hidden layers
        self.hidden1 = nn.Linear(num_states, hidden_dims[0]) # input to hidden, nn.Linear are the next layers after the given input x
        self.hidden2 = nn.Linear(hidden_dims[0]+num_actions, hidden_dims[1]) # hidden to hidden
        # output layer
        self.output = nn.Linear(hidden_dims[1], output_dim) # hidden to output
        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # initialize weights
        self.init_weights()

    def forward(self, x):
        # pull state and action from input
        state, action = x

        # first hidden layer and relu activation
        x = self.hidden1(state)
        x = self.relu(x)

        # in critic (Q-value fn) network, the actions are not included until the second hidden layer
        # feed thru w/ relu activation
        x = self.hidden2(torch.cat([x,action],1))
        x = self.relu(x)
        
        # feed through output layer w/ tanh activation
        x = self.output(x)
        y = self.tanh(x)

        return y
    
    def init_weights(self):
        # init hidden with uniform distribution (-1/sqrt(fan_in), 1/sqrt(fan_in)); fan_in being the input of that particular layer
        # alternative method: nn.init.uniform_(self.hidden1.weight, a=-(1/math.sqrt(self.hidden1.weight.size(1))), b=(1/math.sqrt(self.hidden1.weight.size(1))))
        self.hidden1.weight.data.uniform_(-(1/math.sqrt(self.hidden1.weight.size(1))),(1/math.sqrt(self.hidden1.weight.size(1))))
        self.hidden2.weight.data.uniform_(-(1/math.sqrt(self.hidden2.weight.size(1))),(1/math.sqrt(self.hidden2.weight.size(1))))
        # output layer weights init with uniform distribution (-3e-3,3e-3)
        self.output.weight.data.uniform_(-self.init_weight_limit, self.init_weight_limit)

# [reference] https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py
# Used Udacity tutorial for OU noise generation
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float32)
    
class DDPGAgent:
    def __init__(self, env, params, random_seed) -> None:
        # grabbing parameters
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.actor_lr = params['actor_lr']
        self.critic_lr = params['critic_lr']
        self.batch_size = params['minibatch_size']
        self.buffer_size = params['replay_buffer_size']

        # setting random seed
        self.seed = random.seed(random_seed)

        # setting number of states and actions
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        # choose device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        # initialize critic network w target
        self.critic = Critic(self.num_states, self.num_actions).to(self.device)
        # summary(self.critic, input_size=(2))
        # creating deepcopy to copy the network over to a target
        self.critic_target = deepcopy(self.critic)
        # define optimizer
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        # initialize actor network w target
        self.actor = Actor(self.num_states, self.num_actions).to(self.device)
        # summary(self.actor, input_size=(11,))
        # creating deepcopy to copy the network over to a target
        self.actor_target = deepcopy(self.actor)
        # define optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)

        # OU noise for action selection
        self.noise = OUNoise(self.num_actions, random_seed)

        # initialize replay buffer and prepopulate
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        # self.replay_buffer.prepopulate()

    # get action with some noise
    def get_action(self, state):
        # convert to tensor to feed into network
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        # set to eval mode to not track batch norm
        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state)
            action += self.noise.sample()
        self.actor.train()
        
        return action.numpy()
    
    # updates critic and actor
    def update(self):
        # sample batch
        state_batch, action_batch, reward_batch, next_state_batch, dones_batch = self.replay_buffer.sample_random_minibatch(self.batch_size, self.device)

        # calculate target batch
        with torch.no_grad():
            target = self.calculate_target(reward_batch, next_state_batch)

        # calculate q-value batch
        q_val_batch = self.critic((state_batch, action_batch))

        # update critic by minimizing loss
        loss = nn.MSELoss()

        self.critic_optim.zero_grad()
        loss_val = loss(q_val_batch, target)
        loss_val.backward()        
        self.critic_optim.step()
        
        # using critic to update actor
        loss_actor = -self.critic((state_batch, self.actor(state_batch))) # TODO: should this be negative?
        
        self.actor.zero_grad()
        loss_actor = torch.mean(loss_actor)
        loss_actor.backward()
        self.actor_optim.step()

        # update target network weights
        # update target critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        
        # update target actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        return loss_actor.item()

    def calculate_target(self, reward, next_state):
        next_action = self.actor_target(next_state)
        target = reward + self.gamma*self.critic_target((next_state, next_action))
        return target
    
# TODO: choose device to run on
# main loop
def run_DDPG():
    # initialize parameters
    params = {'actor_lr': 0.0001,
            'critic_lr': 0.001,
            'tau': 0.001,
            'gamma': 0.99,
            'minibatch_size': 64,
            'replay_buffer_size': int(10e6),
            'steps': 100_000}

    # grab cwd for model saving
    cwd = os.getcwd()
    
    # create environment
    env = gym.make('Hopper-v5')

    # ddpg object
    ddpg = DDPGAgent(env, params, random_seed=10)

    # keep track of loss
    scores = []
    scores_deque = deque(maxlen=100)
    loss = []

    # loop through desired number of steps
    for ep in tqdm.tqdm(range(params['steps']), desc="steps"):
        # reset env
        state,_ = env.reset()

        # initialize terminal state
        done = False

        # track cumulative reward
        cumulative_reward = 0

        # while environment isnt terminal
        while not done:
            # grab action with OU noise
            action = ddpg.get_action(state)
            # execute action in env
            next_state, reward, done, truncate, _ = env.step(action)
            # store in buffer
            ddpg.replay_buffer.insert(state, action, reward, next_state, done)

            # learn when buffer reaches batch size
            if len(ddpg.replay_buffer.buffer) > ddpg.batch_size:
                loss_item = ddpg.update()
                loss.append(loss_item)

            # update state
            state = next_state

            # update cumulative reward
            cumulative_reward += reward

        # append to running score and to score deque for average reward approximation
        scores.append(cumulative_reward)
        scores_deque.append(cumulative_reward)

        # save scores
        if ep % 1000 == 0:
            torch.save(ddpg.actor.state_dict(), cwd+'/checkpoint_actor.pth')
            torch.save(ddpg.critic.state_dict(), cwd+'/checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_deque)))   
        
        # reset env if done
        env.reset()

    
    return scores, loss

scores, loss = run_DDPG()

plt.plot(np.arange(1,len(scores)+1), scores)
plt.ylabel('Cumulative reward')
plt.xlabel('Episodes')
plt.show()



# grab cwd for model saving
cwd = os.getcwd()

# instantiate env
env = gym.make('Hopper-v5', render_mode='human')

# initialize parameters
params = {'actor_lr': 0.0001,
        'critic_lr': 0.001,
        'tau': 0.001,
        'gamma': 0.99,
        'minibatch_size': 64,
        'replay_buffer_size': int(10e6),
        'steps': 100_000}

ddpg = DDPGAgent(env, params, random_seed=10)

ddpg.actor.load_state_dict(torch.load(cwd+'/checkpoint_actor.pth'))
ddpg.critic.load_state_dict(torch.load(cwd+'/checkpoint_critic.pth'))

state,_ = env.reset()  
while True:
    action = ddpg.get_action(state)
    env.render()
    next_state, reward, done, truncate, _ = env.step(action)
    state = next_state
    if done:
        break
        
env.close()
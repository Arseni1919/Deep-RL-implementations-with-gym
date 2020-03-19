from IMPORTS import *
import numpy as np
import sys
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from collections import namedtuple, deque, OrderedDict
from copy import copy, deepcopy
import pandas as pd
import time
import shutil
from skimage import transform

def scale_lumininance(obs):
    return np.dot(obs[..., :3], [0.299, 0.587, 0.114])


def resize(obs):
    return transform.resize(obs, (84, 84))


def normalize(obs):
    return obs / 255
    # return (obs - obs.mean()) / np.std(obs)


def preprocess_observation(obs):
    obs_proc = scale_lumininance(obs)
    obs_proc = resize(obs_proc)
    obs_proc = normalize(obs_proc)
    return obs_proc


class DQNAgent:

    def __init__(self, env, network, memory_size=50000,
                 batch_size=32, burn_in=10000, reward_threshold=None,
                 path=None, *args, **kwargs):

        self.env = env
        self.env_name = env.spec.id
        self.network = network
        self.target_network = deepcopy(network)
        self.tau = network.tau
        self.batch_size = batch_size
        self.window = 100
        if reward_threshold is None:
            self.reward_threshold = 195 if 'CartPole' in self.env_name \
                else 300
        else:
            self.reward_threshold = reward_threshold
        self.path = path
        self.timestamp = time.strftime('%Y%m%d_%H%M')
        self.initialize(memory_size, burn_in)

    # Implement DQN training algorithm
    def train(self, epsilon=0.05, gamma=0.99, max_episodes=None,
              batch_size=32, network_sync_frequency=5000, update_freq=4):
        self.gamma = gamma
        self.epsilon = epsilon
        save_freq = 100
        self.fps_buffer = []
        best_ep_avg = 0
        # Populate replay buffer
        print("Populating replay memory buffer...")
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            if done:
                self.s_0 = preprocess_observation(self.env.reset())

        ep = 0
        training = True
        while training:
            self.s_0 = preprocess_observation(self.env.reset())
            self.rewards = 0
            done = False
            first_step = copy(self.step_count)
            ep_start_time = time.time()
            while done == False:
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % update_freq == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())

                if done:
                    ep += 1
                    fps = (self.step_count - first_step) / (time.time() - ep_start_time)
                    self.fps_buffer.append(fps)
                    self.training_rewards.append(self.rewards)
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f} Speed {:.2f} (fps)\t\t".format(
                        ep, mean_rewards, fps), end="")
                    if max(self.mean_training_rewards) > best_ep_avg:
                        best_ep_avg = max(self.mean_training_rewards)
                        print("\nNew best {} episode average: {:.2f}".format(self.window, best_ep_avg))
                        self.save_weights(file_name="ep_{}_dqn_weights.pt".format(ep))
                    if ep % save_freq == 0:
                        # Save rewards
                        self.save_rewards(self.training_rewards[:-save_freq], self.fps_buffer[:-save_freq])
                    if max_episodes is not None:
                        if ep >= max_episodes:
                            training = False
                            print('\nEpisode limit reached.')
                            break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        self.success = True
                        print('\nEnvironment solved in {} steps!'.format(
                            self.step_count))
                        break

    def take_step(self, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            s_0 = np.stack([self.state_buffer])
            action = self.network.get_action(s_0, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        s_1 = preprocess_observation(s_1)
        self.rewards += r
        self.state_buffer.append(self.s_0.copy())
        self.next_state_buffer.append(s_1.copy())
        self.buffer.append(deepcopy(self.state_buffer), action, r, done,
                           deepcopy(self.next_state_buffer))
        self.s_0 = s_1.copy()
        return done

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device)
        actions_t = torch.LongTensor(np.array(actions)).to(
            device=self.network.device).reshape(-1, 1)
        dones_t = torch.BoolTensor(dones).to(device=self.network.device)

        qvals = torch.gather(self.network.get_qvals(states), 1,
                             actions_t).squeeze()
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def initialize(self, memory_size, burn_in):
        self.buffer = experienceReplayBuffer(memory_size, burn_in)
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = preprocess_observation(self.env.reset())
        self.state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        [self.state_buffer.append(np.zeros(self.s_0.size))
         for i in range(self.tau)]
        [self.next_state_buffer.append(np.zeros(self.s_0.size))
         for i in range(self.tau)]
        self.state_buffer.append(self.s_0)
        self.success = False
        if self.path is None:
            self.path = os.path.join(os.getcwd(),
                                     self.env_name, self.timestamp)
        os.makedirs(self.path, exist_ok=True)

    def plot_rewards(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_rewards, label='Rewards')
        plt.plot(self.mean_training_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.ylim([0, np.round(self.reward_threshold) * 1.05])
        plt.savefig(os.path.join(self.path, 'rewards.png'))
        plt.show()

    def save_rewards(self, rewards, speeds):
        file_name = os.path.join(self.path, 'training_rewards.txt')
        if os.path.exists(file_name):
            file = open(file_name, 'a')
        else:
            file = open(file_name, 'w')
            file.write('reward,speed\n')
        for x in zip(rewards, speeds):
            file.write('{},{}\n'.format(x[0], x[1]))
        file.close()

    def save_weights(self, file_name=None):
        if file_name is None:
            file_name = 'dqn_weights.pt'
        weights_path = os.path.join(self.path, file_name)
        torch.save(self.network.state_dict(), weights_path)

    def save_parameters(self, args):
        # Saves .txt file for input parameters
        file = open(os.path.join(self.path, 'parameters.txt'), 'w')
        [file.writelines('\n' + str(k) + ',' + str(v))
         for k, v in vars(args).items()]
        file.close()


class QNetwork(nn.Module):

    def __init__(self, env, learning_rate=1e-3, n_hidden_layers=1,
                 n_hidden_nodes=512, bias=True, activation_function='relu',
                 tau=4, device='cpu', input_dim=(84, 84), *args, **kwargs):
        super(QNetwork, self).__init__()
        self.device = device
        self.actions = np.arange(env.action_space.n)
        self.tau = tau
        n_outputs = env.action_space.n

        activation_function = activation_function.lower()
        if activation_function == 'relu':
            act_func = nn.ReLU()
        elif activation_function == 'tanh':
            act_func = nn.Tanh()
        elif activation_function == 'elu':
            act_func = nn.ELU()
        elif activation_function == 'sigmoid':
            act_func = nn.Sigmoid()
        elif activation_function == 'selu':
            act_func = nn.SELU()

        # CNN modeled off of Mnih et al.
        self.cnn = nn.Sequential(
            nn.Conv2d(tau, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_layer_inputs = self.cnn_out_dim(input_dim)
        # self.fc_layer_inputs = 3136

        # Build a network dependent on the hidden layer and node parameters
        layers = OrderedDict()
        n_layers = 2 * (n_hidden_layers)
        for i in range(n_layers + 1):
            if n_hidden_layers == 0:
                layers[str(i)] = nn.Linear(
                    self.fc_layer_inputs,
                    n_outputs,
                    bias=bias)
            elif i == n_layers:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_outputs,
                    bias=bias)
            elif i % 2 == 0 and i == 0:
                layers[str(i)] = nn.Linear(
                    self.fc_layer_inputs,
                    n_hidden_nodes,
                    bias=bias)
            elif i % 2 == 0 and i < n_layers - 1:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    bias=bias)
            else:
                layers[str(i)] = act_func

        self.fully_connected = nn.Sequential(layers)

        # Set device for GPU's
        if self.device == 'cuda':
            self.cnn.cuda()
            self.fully_connected.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action

    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        state_t = torch.FloatTensor(state).to(device=self.device)
        cnn_out = self.cnn(state_t).reshape(-1, self.fc_layer_inputs)
        return self.fully_connected(cnn_out)

    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.tau, *input_dim)
                        ).flatten().shape[0]


class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size




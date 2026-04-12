import pickle, torch, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy.linalg as LA

class D4rlDataset(Dataset):
    """A simple image dataset class."""
    def __init__(self, data, env_name):
        self.n_episodes = 0

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.not_dones = []

        self.load(data, env_name)
        self.size = len(self.states)

        print('dataset size:', len(self.states))

    def load(self, data, env_name):
        assert('next_observations' in data.keys())
        dataset_size = data['observations'].shape[0]

        for i in range(0, dataset_size):
            self.states.append(data['observations'][i])
            self.next_states.append(data['next_observations'][i])
            self.actions.append(data['actions'][i])
            self.rewards.append([data['rewards'][i]])
            self.not_dones.append([1 - data['terminals'][i]])

        self.states = np.array(self.states)
        self.next_states = np.array(self.next_states)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.not_dones = np.array(self.not_dones)

        self.action_mean = np.mean(self.actions, axis=0)
        self.action_std = np.std(self.actions, axis=0)
        self.state_mean = np.mean(self.states, axis=0)
        self.state_std = np.std(self.states, axis=0)

        self.states = self.normalize_state(self.states)
        self.next_states = self.normalize_state(self.next_states)
        self.actions = self.normalize_action(self.actions)

    def normalize_state(self, state):
        return (state - self.state_mean)/(self.state_std+0.000001)

    def unnormalize_state(self, state):
        return state * (self.state_std+0.000001) + self.state_mean

    def normalize_action(self, action):
        return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        return action * (self.action_std+0.000001) + self.action_mean

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample_idx = idx
        sample = {
            'state': self.states[sample_idx],
            'action': self.actions[sample_idx],
            'next_state': self.next_states[sample_idx],
            'reward': self.rewards[sample_idx],
            'not_done': self.not_dones[sample_idx],
        }

        return sample

"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch, math
import matplotlib.pyplot as plt
from tqdm import tqdm

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, state_dim))
        self.storage['action'] = np.zeros((max_size, action_dim))
        self.storage['next_state'] = np.zeros((max_size, state_dim))
        self.storage['reward'] = np.zeros((max_size, 1))
        self.storage['not_done'] = np.zeros((max_size, 1))

        self.min_r, self.max_r = 0, 0

        self.action_mean = None
        self.action_std = None
        self.action_s = None
        self.state_mean = None
        self.state_std = None
        self.state_s = None

    def add(self, state, action, next_state, reward, done):
        self.size = min(self.size + 1, self.max_size)

        self.storage['state'][self.ptr] = state.copy()
        self.storage['action'][self.ptr] = action.copy()
        self.storage['next_state'][self.ptr] = next_state.copy()

        if self.action_mean is not None:
            new_state_mean = self.state_mean + (state - self.state_mean) / self.size
            new_state_s = self.state_s + (state - self.state_mean) * (state - new_state_mean)
            new_action_mean = self.action_mean + (action - self.action_mean) / self.size
            new_action_s = self.action_s + (action - self.action_mean) * (action - new_action_mean)

            self.state_mean = new_state_mean
            self.state_s = new_state_s
            self.state_std = np.sqrt(new_state_s / (self.size - 1))
            self.action_mean = new_action_mean
            self.action_s = new_action_s
            self.action_std = np.sqrt(new_action_s / (self.size - 1))

            self.min_r = min(self.min_r, reward)
            self.max_r = max(self.max_r, reward)

        self.storage['reward'][self.ptr] = reward
        self.storage['not_done'][self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.normalize_state(self.storage['state'][ind])).to(self.device),
            torch.FloatTensor(self.normalize_action(self.storage['action'][ind])).to(self.device),
            torch.FloatTensor(self.normalize_state(self.storage['next_state'][ind])).to(self.device),
            torch.FloatTensor(self.storage['reward'][ind]).to(self.device),
            torch.FloatTensor(self.storage['not_done'][ind]).to(self.device),
        )

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def normalize_state(self, state):
        return (state - self.state_mean)/(self.state_std+0.000001)

    def unnormalize_state(self, state):
        return state * (self.state_std+0.000001) + self.state_mean

    def normalize_action(self, action):
        return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        return action * (self.action_std+0.000001) + self.action_mean

    def load(self, data):
        assert('next_observations' in data.keys())

        for i in range(data['observations'].shape[0]):
            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                     data['rewards'][i], data['terminals'][i])
                     
        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.action_s = self.action_std * self.action_std * (self.size + 1)

        self.state_mean = np.mean(self.storage['state'][:self.size], axis=0)
        self.state_std = np.std(self.storage['state'][:self.size], axis=0)
        self.state_s = self.state_std * self.state_std * (self.size + 1)

        self.min_r, self.max_r = self.storage['reward'].min(), self.storage['reward'].max()

        # print(self.state_mean, self.state_std, self.action_mean, self.action_std)
        print("Dataset size:" + str(self.size))
        print(self.min_r, self.max_r)
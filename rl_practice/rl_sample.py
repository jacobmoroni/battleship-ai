"""
This is a sample of a RL network used to solve the lunar lander problem
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm


class PolicyNetwork(nn.Module):
    """
    Policy Network for rl
    """

    def __init__(self, state_dimension, action_dimension):
        super(PolicyNetwork, self).__init__()
        # self.policy_net = nn.Sequential(nn.Linear(state_dimension, 10),
        #                                 nn.ReLU(),
        #                                 nn.Linear(10, 10),
        #                                 nn.ReLU(),
        #                                 nn.Linear(10, 10),
        #                                 nn.ReLU(),
        #                                 nn.Linear(10, action_dimension))
        self.policy_net = nn.Sequential(nn.Linear(state_dimension, 128),
                                        nn.Dropout(p=0.6),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.Dropout(p=0.6),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.Dropout(p=0.6),
                                        nn.ReLU(),
                                        nn.Linear(128, action_dimension))
        self.policy_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Push state through the network
        """
        scores = self.policy_net(x)
        return self.policy_softmax(scores)


class ValueNetwork(nn.Module):
    """
    Value Network Class
    """

    def __init__(self, state_dimension):
        super(ValueNetwork, self).__init__()
        self.value_net = nn.Sequential(nn.Linear(state_dimension, 10),
                                       nn.ReLU(),
                                       nn.Linear(10, 10),
                                       nn.ReLU(),
                                       nn.Linear(10, 10),
                                       nn.ReLU(),
                                       nn.Linear(10, 1))

    def forward(self, x):
        """
        push state through value net
        """
        return self.value_net(x)


class AdvantageDataset(Dataset):
    """
    Dataset for advantage
    """

    def __init__(self, experience):
        super(AdvantageDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp[0], chosen_exp[4]

    def __len__(self):
        return self._length


class PolicyDataset(Dataset):
    """
    Dataset for policy
    """

    def __init__(self, experience):
        super(PolicyDataset, self).__init__()
        self._exp = experience
        self._num_runs = len(experience)
        self._length = reduce(lambda acc, x: acc + len(x), experience, 0)

    def __getitem__(self, index):
        idx = 0
        seen_data = 0
        current_exp = self._exp[0]
        while seen_data + len(current_exp) - 1 < index:
            seen_data += len(current_exp)
            idx += 1
            current_exp = self._exp[idx]
        chosen_exp = current_exp[index - seen_data]
        return chosen_exp

    def __len__(self):
        return self._length


def calculateReturns(trajectories, gamma):
    """
    calculates returns 
    """
    for i, trajectory in enumerate(trajectories):
        current_reward = 0
        for j in reversed(range(len(trajectory))):
            state, probs, action_index, reward = trajectory[j]
            ret = reward + gamma * current_reward
            trajectories[i][j] = (state, probs, action_index, reward, ret)
            current_reward = ret


def calculateAdvantages(trajectories, value_net):
    """
    Calculates advantages
    """
    for i, trajectory in enumerate(trajectories):
        for j, exp in enumerate(trajectory):
            advantage = exp[4] - value_net(torch.from_numpy(exp[0]).float().unsqueeze(0))[0, 0].detach().numpy()
            trajectories[i][j] = (exp[0], exp[1], exp[2], exp[3], exp[4], advantage)


def renderNow(env, policy, num_actions, states):
    """
    Render environment
    """

    env.reset()
    for i in range(1000):

        env.render()
        action = policy(torch.from_numpy(states).float().view(1, -1))
        ac = action.detach().numpy().reshape((num_actions))
        action_index = np.random.choice(range(num_actions), p=ac)
        _, _, end_now, _ = env.step(action_index)
        if end_now:
            break


def plotResults(reward_list, loss_list):
    """
    Plot results
    """
    plt.figure()
    plt.plot(reward_list)
    plt.title("Average Reward over Time")
    plt.xlabel("Time (epochs)")
    plt.ylabel("Reward")

    plt.figure()
    plt.plot(loss_list)
    plt.title("Loss over time")
    plt.xlabel("Time (epochs)")
    plt.ylabel("Loss")
    plt.show()

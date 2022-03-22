"""
script to run RL for lunar landar v2
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
import numpy as np
# from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_sample import *

env = gym.make('LunarLander-v2')
num_states = 8
num_actions = 4
policy = PolicyNetwork(num_states, num_actions)
value = ValueNetwork(num_states)

# policy_optim = optim.Adam(policy.parameters(), lr=1e-2, weight_decay=0.01)
policy_optim = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=0.01)
value_optim = optim.Adam(value.parameters(), lr=1e-3, weight_decay=1)
value_criteria = nn.MSELoss()

# Hyperparameters
epochs = 200  # 1000
env_samples = 100
episode_length = 200  # 4000
# gamma = 0.9
gamma = 0.99
value_epochs = 2
policy_epochs = 5
batch_size = 32
policy_batch_size = 256
epsilon = 0.2
standing_time_list = []
loss_list = []
reward_list = []

loop = tqdm(total=epochs, position=0, leave=False)

for epoch in range(epochs):

    # generate rollouts
    rollouts = []
    standing_length = 0
    max_x_total = 0
    avg_reward = 1
    for _ in range(env_samples):
        current_rollout = []
        s = env.reset()
        max_x = -10
        min_x = 0
        for i in range(episode_length):
            action = policy(torch.from_numpy(s).float().view(1, -1))
            ac = action.detach().numpy().reshape((num_actions))
            action_index = np.random.choice(range(num_actions), p=ac)
            s_prime, r, t, _ = env.step(action_index)

            avg_reward = avg_reward + r
            current_rollout.append((s, action.detach().reshape(-1), action_index, r))
            standing_length += 1

            if t:
                break

            s = s_prime
        max_x_total += max_x
        rollouts.append(current_rollout)

    avg_max_x = max_x_total / env_samples
    avg_standing_time = standing_length / env_samples
    avg_reward = avg_reward/env_samples
    standing_time_list.append(avg_standing_time)
    reward_list.append(avg_reward)
    calculateReturns(rollouts, gamma)

    # Approximate the value function
    value_dataset = AdvantageDataset(rollouts)
    value_loader = DataLoader(value_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    for _ in range(value_epochs):
        # train value network
        total_loss = 0
        for state, returns in value_loader:
            value_optim.zero_grad()
            returns = returns.unsqueeze(1).float()
            expected_returns = value(state.float())
            loss = value_criteria(expected_returns, returns)
            total_loss += loss.item()
            loss.backward()
            value_optim.step()
        loss_list.append(total_loss)

    calculateAdvantages(rollouts, value)

    # Learn a policy
    policy_dataset = PolicyDataset(rollouts)
    policy_loader = DataLoader(policy_dataset, batch_size=policy_batch_size, shuffle=True, pin_memory=True)
    for _ in range(policy_epochs):
        # train policy network
        for state, probs, action_index, reward, ret, advantage in policy_loader:
            policy_optim.zero_grad()
            current_batch_size = reward.size()[0]
            advantage = advantage.detach().float()  # ret.float() #
            p = policy(state.float())
            ratio = p[range(current_batch_size), action_index] / probs[range(current_batch_size), action_index]

            lhs = ratio * advantage
            rhs = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
            loss = -torch.mean(torch.min(lhs, rhs))

            loss.backward()
            policy_optim.step()

    if epoch % 50 == 0 & epoch != 0:
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

    loop.set_description('epoch:{},loss:{:.4f},avg_reward:{:.5f}'.format(epoch, total_loss, avg_reward))
    loop.update(1)
    if avg_reward > 200:
        break

loop.close()
plotResults()
renderNow()

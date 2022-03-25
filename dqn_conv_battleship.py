"""
Deep Q Network built to solve the lunar lander v2 problem
"""
import random
from collections import deque, namedtuple
import torch
import numpy as np
import torch.nn as nn
from ipdb import set_trace
from torch.nn import functional as F
from torch import optim


class HyperParameters():
    """
    Class to store hyper parameters
    """

    def __init__(self):
        self.buffer_size = int(1e5)  # Replay memory size
        self.batch_size = 64         # Number of experiences to sample from memory
        self.gamma = 0.99            # Discount factor
        self.tau = 1e-3              # Soft update parameter for updating fixed q network
        self.learning_rate = 1e-4    # Q Network learning rate
        self.q_update_freq = 4       # How often to update Q network
        self.max_episodes = 2000     # Max number of episodes to play
        self.max_steps = 1000        # Max steps allowed in a single episode/play
        self.env_solved = 200        # Max score at which we consider environment to be solved

        # Epsilon schedule
        self.eps_start = 1.0         # Default/starting value of eps
        self.eps_decay = 0.999       # Epsilon decay rate
        self.eps_min = 0.01          # Minimum epsilon

        # Random seed
        self.seed = 0


class ConvQNetwork(nn.Module):
    """
    DQN that uses 2d convolutions
    """

    def __init__(self, h, w, outputs, seed=None):
        super(ConvQNetwork, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.action_space = outputs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2dSizeOut(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2dSizeOut(conv2dSizeOut(conv2dSizeOut(w)))
        convh = conv2dSizeOut(conv2dSizeOut(conv2dSizeOut(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        Forward step
        """

        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_1d = self.head(x.view(x.size(0), -1))
        return x_1d.view(-1, 2, int(self.action_space/2))


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    """
    Replay memory class for convolutional DQN
    """

    def __init__(self, hyperparms):
        self.hp = hyperparms
        self.memory = deque([], maxlen=self.hp.buffer_size)

    def add(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        """Sample the memory"""
        return random.sample(self.memory, self.hp.batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    DQN Agent interacts with the environment,
    stores the experience and learns from it

    Parameters
    ----------
    state_size (int): Dimension of state
    action_size (int): Dimension of action
    seed (int): random seed
    """

    def __init__(self, state_w, state_h, action_size, hyperparams):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hp = hyperparams
        self.action_size = action_size
        # self.seed = random.seed(seed)
        random.seed(self.hp.seed)
        # Initialize Q and Fixed Q networks
        self.policy_net = ConvQNetwork(state_w, state_h, action_size, self.hp.seed).to(self.device)
        self.target_net = ConvQNetwork(state_w, state_h, action_size, self.hp.seed).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        # Initilize memory
        self.memory = ReplayMemory(self.hp)
        self.timestep = 0

    def optimizeModel(self):
        """Optimize the model"""

        if len(self.memory) < self.hp.batch_size:
            return
        transitions = self.memory.sample()
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(np.invert(batch.done),
                                      device=self.device, dtype=torch.bool)
        # non_final_next_states = torch.cat(batch.next_state[non_final_mask])
        non_final_next_states = torch.tensor(np.array(batch.next_state))
        state_batch = torch.tensor(np.array(batch.state))
        action_batch = torch.tensor(np.array(batch.action))
        reward_batch = torch.tensor(np.array(batch.reward))

        cuda_state_batch = state_batch.float().unsqueeze(1).to(self.device)
        state_action_values = self.policy_net.forward(cuda_state_batch)
        state_action_values = state_action_values.gather(2, action_batch.unsqueeze(2).to(self.device))
        next_state_values = torch.zeros(self.hp.batch_size, device=self.device)
        # next_state_values[non_final_mask] = self.target_net.forward(
        #     non_final_next_states.float().unsqueeze(1).to(self.device)).max(1)[0].detach()
        next_state_values = self.target_net.forward(
            non_final_next_states.float().unsqueeze(1).to(self.device))
        actions = next_state_values.squeeze().cpu().data.numpy()
        choices_remaining_mask = state_batch != 0
        row_remaining_mask = choices_remaining_mask.sum(2) == state_batch.shape[1]
        row_mask_array = np.ma.array(actions[:, 0], mask=row_remaining_mask)
        row = np.argmax(row_mask_array, 1)
        row_vals = np.max(row_mask_array, 1).data
        col_remaining_mask = choices_remaining_mask[torch.arange(choices_remaining_mask.size(0)), row]
        col_vals = np.max(np.ma.array(actions[:, 1], mask=col_remaining_mask), 1).data
        next_state_values = np.stack((row_vals, col_vals)).transpose()

        expected_state_action_values = (torch.tensor(next_state_values).to(
            self.device) * self.hp.gamma) + reward_batch.unsqueeze(1).to(self.device)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(2))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

    def step(self, state, action, reward, next_state, done):
        """
        Update Agent's knowledge

        Parameters
        ----------
        state (array_like): Current state of environment
        action (int): Action taken in current state
        reward (float): Reward received after taking action
        next_state (array_like): Next state returned by the environment after taking action
        done (bool): whether the episode ended after taking action
        """
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        # if self.timestep % self.hp.q_update_freq == 0:
        #     if len(self.memory) > self.hp.batch_size:
        #         sampled_experiences = self.memory.sample()
        #         self.learn(sampled_experiences)

    # def learn(self, experiences):
    #     """
    #     Learn from experience by training the q_network

    #     Parameters
    #     ----------
    #     experiences (array_like): List of experiences sampled from agent's memory
    #     """
    #     states, actions, rewards, next_states, dones = Transition(*zip(*experiences))
    #     # Get the action with max Q value
    #     action_values = self.fixed_network.forward(next_states).detach()
    #     # Notes
    #     # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
    #     # unsqueeze operation --> np.reshape
    #     # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
    #     max_action_values = action_values.max(1)[0].unsqueeze(1)

    #     # If done just use reward, else update Q_target with discounted action values
    #     q_target = rewards + (self.hp.gamma * max_action_values * (1 - dones))
    #     q_expected = self.q_network.forward(states).gather(1, actions)

    #     # Calculate loss
    #     loss = F.mse_loss(q_expected, q_target)
    #     self.optimizer.zero_grad()
    #     # backward pass
    #     loss.backward()
    #     # update weights
    #     self.optimizer.step()

    #     # Update fixed weights
    #     self.updateFixedNetwork(self.q_network, self.fixed_network)

    # def updateFixedNetwork(self, q_network, fixed_network):
    #     """
    #     Update fixed network by copying weights from Q network using TAU param

    #     Parameters
    #     ----------
    #     q_network (PyTorch model): Q network
    #     fixed_network (PyTorch model): Fixed target network
    #     """
    #     for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
    #         target_parameters.data.copy_(self.hp.tau * source_parameters.data +
    #                                      (1.0 - self.hp.tau) * target_parameters.data)

    def selectAction(self, state, eps=0.0):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        choices_remaining_mask = state != 0
        row_remaining_mask = choices_remaining_mask.sum(1) == state.shape[0]
        if rnd < eps:
            row_prob = np.invert(row_remaining_mask)/np.invert(row_remaining_mask).sum()
            row = int(np.random.choice(np.arange(self.action_size/2), p=row_prob))
            col_prob = np.invert(choices_remaining_mask[row])/np.invert(choices_remaining_mask[row]).sum()
            col = int(np.random.choice(np.arange(self.action_size/2), p=col_prob))
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(1).to(self.device)
            # state = torch.from_numpy(state).float().to(self.device)
            # set the network into evaluation mode
            # self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net.forward(state)

                # action = action_unfilt.max(1)[1].view(1, 1)
            # Back to training mode
            # self.policy_net.train()
            actions = action_values.squeeze().cpu().data.numpy()
            row = np.argmax(np.ma.array(actions[0], mask=row_remaining_mask))
            col = np.argmax(np.ma.array(actions[1], mask=choices_remaining_mask[row]))
        return (row, col)

    def checkpoint(self, filename):
        """
        Saves the current model to load later
        """
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        """
        Load saved data
        """
        self.policy_net.load_state_dict(torch.load(filename))

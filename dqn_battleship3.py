"""
Deep Q Network built to solve the lunar lander v2 problem
"""
import random
from collections import deque, namedtuple
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from ipdb import set_trace


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
        return self.head(x.view(x.size(0), -1))


class ReplayBuffer:
    """
    Replay memory allow agent to record experiences and learn from them

    Parametes
    ---------
    buffer_size (int): maximum size of internal memory
    batch_size (int): sample size from experience
    seed (int): random seed
    """

    def __init__(self, hyperparams):

        self.hp = hyperparams
        random.seed(self.hp.seed)
        self.memory = deque(maxlen=self.hp.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors
        """
        experiences = random.sample(self.memory, k=self.hp.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(np.stack([experience.state
                                            for experience in experiences
                                            if experience is not None])).float().to(self.device)
        actions = torch.from_numpy(np.stack([experience.action
                                             for experience in experiences
                                             if experience is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.stack([experience.reward
                                             for experience in experiences
                                             if experience is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([experience.next_state
                                                 for experience in experiences
                                                 if experience is not None])).float().to(self.device)
        # Convert done from boolean to int
        dones = torch.from_numpy(np.stack([experience.done for
                                           experience in experiences
                                           if experience is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

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
        self.optimizer = optim.Adam(self.policy_net.parameters())
        # Initilize memory
        self.memory = ReplayBuffer(self.hp)
        self.timestep = 0

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
        if self.timestep % self.hp.q_update_freq == 0:
            if len(self.memory) > self.hp.batch_size:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def learn(self, experiences):
        """
        Learn from experience by training the policy_net

        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.target_net.forward(next_states.unsqueeze(1)).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        q_target = rewards.unsqueeze(1) + (self.hp.gamma * max_action_values * (1 - dones.unsqueeze(1)))

        q_expected = self.policy_net.forward(states.unsqueeze(1)).gather(1, actions.unsqueeze(1))

        # Calculate loss
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # Update fixed weights
        self.updateFixedNetwork(self.policy_net, self.target_net)

    def updateFixedNetwork(self, policy_net, target_net):
        """
        Update fixed network by copying weights from Q network using TAU param

        Parameters
        ----------
        policy_net (PyTorch model): Q network
        target_net (PyTorch model): Fixed target network
        """
        for source_parameters, target_parameters in zip(policy_net.parameters(), target_net.parameters()):
            target_parameters.data.copy_(self.hp.tau * source_parameters.data +
                                         (1.0 - self.hp.tau) * target_parameters.data)

    def act(self, state, eps=0.0):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        choices_remaining = state.flatten() == 0

        if rnd < eps:
            return np.random.choice(np.arange(self.action_size), p=choices_remaining/choices_remaining.sum())
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(1).to(self.device)
            # set the network into evaluation mode
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net.forward(state)
            # Back to training mode
            self.policy_net.train()
            action_mask = np.invert(choices_remaining)
            action = np.argmax(np.ma.array(action_values.cpu().data.numpy(), mask=action_mask))
            return action

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

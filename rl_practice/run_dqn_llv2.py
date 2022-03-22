"""
Main run file for DQN
"""
import sys
from time import time
from collections import deque
import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn_lunar_lander import DQNAgent, HyperParameters

# HyperParameters and global variables

BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
UPDATE_EVERY = 4        # How often to update Q network
MAX_EPISODES = 2000  # Max number of episodes to play
MAX_STEPS = 1000     # Max steps allowed in a single episode/play
ENV_SOLVED = 200     # MAX score at which we consider environment to be solved
PRINT_EVERY = 100    # How often to print the progress

# Epsilon schedule
EPS_START = 1.0      # Default/starting value of eps
EPS_DECAY = 0.999    # Epsilon decay rate
EPS_MIN = 0.01       # Minimum epsilon


class RLTrainer:
    """
    class to run training for RL
    """

    def __init__(self, hyperparams):
        self.hp = hyperparams
        self.env = gym.make('LunarLander-v2')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.dqn_agent = DQNAgent(self.state_size, self.action_size, self.hp)
        self.env.reset(seed=0)

        self.scores = []
        # Maintain a list of last 100 scores
        self.scores_window = deque(maxlen=100)
        self.eps = EPS_START

    def plotResults(self):
        """
        Plot scores over time
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.scores)
        # A bit hard to see the above plot, so lets smooth it (red)
        plt.plot(pd.Series(self.scores).rolling(100).mean())
        plt.title('DQN Training')
        plt.xlabel('# of episodes')
        plt.ylabel('score')
        plt.show()

    def renderRun(self):
        """
        Renders a run with the current model
        """
        self.env = gym.make('LunarLander-v2')
        score = 0
        state = self.env.reset()
        while True:
            self.env.render()
            action = self.dqn_agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.dqn_agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

    def loadAgent(self, file):
        """
        Load anent from file
        """
        self.dqn_agent.load(file)

    def closeRender(self):
        """
        close active render
        """
        self.env.close()

    def runLoop(self, num_episodes):
        """
        Runs loop to continue training
        """
        start = time()

        loop = tqdm(total=num_episodes, position=0, leave=False)

        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            score = 0
            for _ in range(self.hp.max_steps):
                action = self.dqn_agent.act(state, self.eps)
                next_state, reward, done, info = self.env.step(action)
                self.dqn_agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

                self.eps = max(self.eps * self.hp.eps_decay, self.hp.eps_min)
                if episode % PRINT_EVERY == 0:
                    mean_score = np.mean(self.scores_window)
                    res = f'\r Progress {episode}/{MAX_EPISODES}, average score:{mean_score:.2f}'
                    print(res, end="")
                if score >= ENV_SOLVED:
                    mean_score = np.mean(self.scores_window)
                    print(f'\rEnvironment solved in {episode} episodes, average score: {mean_score:.2f}', end="")
                    sys.stdout.flush()
                    self.dqn_agent.checkpoint('solved_200.pth')
                    break

            self.scores_window.append(score)
            self.scores.append(score)
            loop.set_description(f'episode:{episode}, avg_reward:{np.mean(self.scores_window):.5f}')
            loop.update(1)

        end = time()
        print(f'Took {end-start} seconds')


if __name__ == '__main__':
    hp = HyperParameters()
    hp.buffer_size = BUFFER_SIZE
    hp.batch_size = BATCH_SIZE
    hp.gamma = GAMMA
    hp.tau = TAU
    hp.learning_rate = LR
    hp.q_update_freq = UPDATE_EVERY
    hp.max_episodes = MAX_EPISODES
    hp.max_steps = MAX_STEPS
    hp.env_solved = ENV_SOLVED
    hp.eps_start = EPS_START
    hp.eps_decay = EPS_DECAY
    hp.eps_min = EPS_MIN
    hp.seed = 0

    trainer = RLTrainer(hp)
    # trainer.runLoop(hyperparams.max_episodes)

    # env = gym.make('LunarLander-v2')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # dqn_agent = DQNAgent(state_size, action_size, hyperparams)
    # env.reset(seed=0)

    # start = time()

    # scores = []
    # # Maintain a list of last 100 scores
    # scores_window = deque(maxlen=100)
    # EPS = EPS_START

    # loop = tqdm(total=MAX_EPISODES, position=0, leave=False)

    # for episode in range(1, MAX_EPISODES + 1):
    #     state = env.reset()
    #     SCORE = 0
    #     for t in range(MAX_STEPS):
    #         action = dqn_agent.act(state, EPS)
    #         next_state, reward, done, info = env.step(action)
    #         dqn_agent.step(state, action, reward, next_state, done)
    #         state = next_state
    #         SCORE += reward
    #         if done:
    #             break

    #         EPS = max(EPS * EPS_DECAY, EPS_MIN)
    #         if episode % PRINT_EVERY == 0:
    #             mean_score = np.mean(scores_window)
    #             res = f'\r Progress {episode}/{MAX_EPISODES}, average score:{mean_score:.2f}'
    #             print(res, end="")
    #         if SCORE >= ENV_SOLVED:
    #             mean_score = np.mean(scores_window)
    #             print(f'\rEnvironment solved in {episode} episodes, average score: {mean_score:.2f}', end="")
    #             sys.stdout.flush()
    #             dqn_agent.checkpoint('solved_200.pth')
    #             break

    #     scores_window.append(SCORE)
    #     scores.append(SCORE)
    #     loop.set_description(f'episode:{episode}, avg_reward:{np.mean(scores_window):.5f}')
    #     loop.update(1)

    # end = time()
    # print(f'Took {end-start} seconds')

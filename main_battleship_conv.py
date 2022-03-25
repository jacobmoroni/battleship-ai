"""
Main run file for DQN
"""
# import sys
from time import time, sleep
from collections import deque
import curses
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn_conv_battleship import DQNAgent, HyperParameters
from battleship_conv_wrapper import BattleshipEnvironment

# HyperParameters

BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
UPDATE_EVERY = 10        # How often to update Q network
MAX_EPISODES = 2000     # Max number of episodes to play
MAX_STEPS = 100         # Max number of steps that can be taken in an episode

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
        self.env = BattleshipEnvironment()
        self.action_size = self.env.num_actions
        self.dqn_agent = DQNAgent(self.env.state_w, self.env.state_h, self.action_size, self.hp)
        self.env.reset(seed=0)
        self.render_output = None

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

    def renderState(self, state, action, reward, score):
        """
        Renders a single state
        """
        board_size_dim = self.env.board_size
        # board_size_full = board_size_dim**2
        board = state
        self.render_output.addstr(0, 0, str(board))
        self.render_output.addstr(
            11, 0, f"action: {action}, reward: {reward}, score: {score}")
        self.render_output.refresh()

    def renderRun(self):
        """
        Renders a run with the current model
        """
        self.render_output = curses.initscr()
        curses.noecho()
        curses.cbreak()
        print("\n\n\n\n\n\n\n\n\n\n\n\n")

        # self.env = BattleshipEnvironment()
        score = 0
        state = self.env.reset()
        data = {"action": [],
                "reward": [],
                "score": [],
                "ship_state": []}
        # for i in range(100):
        while True:
            # action = i  # self.dqn_agent.act(state)
            action = self.dqn_agent.selectAction(state)
            next_state, reward, done, _ = self.env.step(action)
            self.dqn_agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            data['action'].append(action)
            data['reward'].append(reward)
            data['score'].append(score)
            # data['ship_state'].append(state[100:])
            try:
                self.renderState(state, action, reward, score)
                sleep(0.5)
            finally:
                curses.echo()
                curses.nocbreak()
                curses.endwin()

            if done:
                break
        return data

    def loadAgent(self, file):
        """
        Load anent from file
        """
        self.dqn_agent.load(file)

    def runLoop(self, num_episodes):
        """
        Runs loop to continue training
        """
        start = time()

        loop = tqdm(total=num_episodes, position=0, leave=False)

        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            # self.env.battleship.printShips()
            # from ipdb import set_trace
            # set_trace()
            score = 0
            for _ in range(self.hp.max_steps):
                action = self.dqn_agent.selectAction(state, self.eps)
                next_state, reward, done, _ = self.env.step(action)
                self.dqn_agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                self.dqn_agent.optimizeModel()
                if done:
                    break

                self.eps = max(self.eps * self.hp.eps_decay, self.hp.eps_min)
                # if score >= ENV_SOLVED:
                #     mean_score = np.mean(self.scores_window)
                #     print(f'\rEnvironment solved in {episode} episodes, average score: {mean_score:.2f}', end="")
                #     sys.stdout.flush()
                #     self.dqn_agent.checkpoint('solved_200.pth')
                #     break
            if episode % self.hp.target_update_freq == 0:
                self.dqn_agent.target_net.load_state_dict(self.dqn_agent.policy_net.state_dict())
            self.scores_window.append(score)
            self.scores.append(score)
            loop.set_description(
                f'episode:{episode}, avg_score:{np.mean(self.scores_window):.5f}, last_score: {score}, turns: {self.env.turns_taken}')
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
    hp.target_update_freq = UPDATE_EVERY
    hp.max_episodes = MAX_EPISODES
    hp.max_steps = MAX_STEPS
    # hp.env_solved = ENV_SOLVED
    hp.eps_start = EPS_START
    hp.eps_decay = EPS_DECAY
    hp.eps_min = EPS_MIN
    hp.seed = 0

    trainer = RLTrainer(hp)
    trainer.runLoop(hp.max_episodes)

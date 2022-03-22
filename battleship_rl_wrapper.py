"""
Battleship Reinforcement Learning wrapper
"""
import numpy as np
from battleship import BattleshipBoard


class BattleshipEnvironment:
    """
    Environment interface for RL with battleship game
    """

    def __init__(self):
        self.board_size = 10
        self.consecutive_hit_boost = 0
        self.num_ships = 5
        self.battleship = BattleshipBoard(self.board_size, [])
        self.state = np.zeros(self.board_size**2 + self.num_ships)
        self.num_actions = self.board_size**2
        self.ship_states = self.battleship.active_ships

    def reset(self, seed=-1):
        """
        resets the environment with an optional seed value
        """
        self.battleship.reset(seed)
        self.state = np.zeros(self.board_size**2 + self.num_ships)
        return self.state

    def actionToCoord(self, action):
        """
        converts the action int to a fire coordinate
        """
        row = action//self.board_size
        col = action % self.board_size
        return row, col

    def calculateReward(self, valid, hit, sunk):
        """
        Calculates the reward for a turn given the status of the last fire
        """
        reward = 0
        if not valid:
            return -50
        if sunk:
            reward += 50
        if hit:
            reward += 10
            if self.consecutive_hit_boost > 0:
                reward += self.consecutive_hit_boost
            self.consecutive_hit_boost = 4
        else:
            self.consecutive_hit_boost -= 1
            reward -= 1
        return reward

    def step(self, action):
        """
        Step through the game one step with an action
        """
        row, col = self.actionToCoord(action)
        valid, hit, sunk = self.battleship.fire(row, col)
        reward = self.calculateReward(valid, hit, sunk)
        next_state = np.hstack((self.battleship.grid.flatten(), self.battleship.active_ships))
        done = np.sum(self.battleship.activeShips) == 0
        info = ""
        return next_state, reward, done, info

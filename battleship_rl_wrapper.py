"""
Battleship Reinforcement Learning wrapper
"""
from random import choice
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
        self.battleship = BattleshipBoard(self.board_size, 0)
        self.num_actions = self.board_size**2
        self.ship_states = self.battleship.active_ships
        self.state_size = self.board_size**2
        self.turns_taken = 0
        self.prev_hit = False
        self.prev_action = None

    def reset(self, seed=None):
        """
        resets the environment with an optional seed value
        """
        self.battleship.reset(seed)
        state = np.zeros(self.board_size**2)
        self.turns_taken = 0
        self.prev_hit = False
        self.prev_action = None
        return state

    def actionToCoord(self, action):
        """
        converts the action int to a fire coordinate
        """
        row = action//self.board_size
        col = action % self.board_size
        return row, col

    def getLinearHits(self, coord, direction):
        """
        Get linear hits
        """
        counter = 0
        idx = 1
        if direction == 0:
            while (coord[0]+idx < self.board_size and self.battleship.shots[coord[0]+idx, coord[1]] == 1):
                counter += 1
                idx += 1
        if direction == 1:
            while (coord[0]-idx >= 0 and self.battleship.shots[coord[0]-idx, coord[1]] == 1):
                counter += 1
                idx += 1
        if direction == 2:
            while (coord[1]+idx < self.board_size and self.battleship.shots[coord[0], coord[1]+idx] == 1):
                counter += 1
                idx += 1
        if direction == 3:
            while (coord[1]-idx >= 0 and self.battleship.shots[coord[0], coord[1]-idx] == 1):
                counter += 1
                idx += 1
        return counter

    def getAdjacentHits(self, action):
        """
        find number of adjacent hits
        """
        coord = self.actionToCoord(action)
        counter = 0
        counter += self.getLinearHits(coord, 0)
        counter += self.getLinearHits(coord, 1)
        counter += self.getLinearHits(coord, 2)
        counter += self.getLinearHits(coord, 3)
        return counter

    def getNextState(self, action, sunk):
        """
        Gets the next state. If a ship is sunk remove hits from board
        """
        if sunk:
            action_coord = self.actionToCoord(action)
            ship = self.battleship.shots[action_coord]
            self.battleship.shots[np.where(self.battleship.ships == ship-1)] = ship
        return self.battleship.shots.flatten()

    def calculateReward(self, valid, hit, sunk, action):
        """
        Calculates the reward for a turn given the status of the last fire
        """
        reward = 0
        if not valid:
            return -500
        if sunk:
            reward += 50
        if hit:
            reward += 10
            if self.consecutive_hit_boost > 0:
                reward += self.consecutive_hit_boost
            self.consecutive_hit_boost = 10
        else:
            self.consecutive_hit_boost //= 2
            reward -= 1
        reward += self.getAdjacentHits(action)*5
        return reward

    def step(self, action):
        """
        Step through the game one step with an action
        """
        row, col = self.actionToCoord(action)
        self.turns_taken += 1
        valid, hit, sunk = self.battleship.fire(row, col)
        reward = self.calculateReward(valid, hit, sunk, action)
        next_state = self.getNextState(action, sunk)
        if not valid:
            done = True
        else:
            done = np.sum(self.battleship.active_ships) == 0
        info = ""
        return next_state, reward, done, info

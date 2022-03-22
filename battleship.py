"""
Battleship game class
"""
from enum import Enum
import numpy as np
from random import randint, choice, seed

class Orientation(Enum):
    """
    Orientation enum for directionality when placing ships
    """
    VERTICAL = 1
    HORIZONTAL = 2


class BattleshipBoard:
    """
    BattleshipBoard class
    """

    def __init__(self, grid_size, random_seed=None):
        self.shots_size = grid_size
        self.shots = np.zeros(shape=(grid_size, grid_size), dtype=np.int8)
        self.ships = np.zeros(shape=(grid_size, grid_size), dtype=np.int8)
        self.max_hits_per_ship = []
        self.hits_per_ship = []
        self.active_ships = []
        seed(random_seed)
            
    def fire(self, row, col):
        """
        fire at the desired row and column
        Updates the board to reflect hit or not
        Returns:
            (new fire, hit ship, sunk ship)
        """
        hit = False
        sunk = False 
        if self.shots[row,col] != 0:
            return False, False, False
        guess = self.ships[row,col]
        if guess == 0:
            self.shots[row,col] = -1
            return True, False, False
        else:
            self.shots[row,col] = 1
            print(guess)
            self.hits_per_ship[guess-1] += 1
            if self.hits_per_ship[guess-1] == self.max_hits_per_ship[guess-1]:
                self.active_ships[guess-1] = False
                sunk = True
            return True,True, sunk
   
    def printShots(self):
        """
        Prints the current state of the board
        """
        print(self.shots)

    def printShips(self):
        print(self.ships)

    def placeShip(self, row, col, orientation, size):
        """
        Places ship of specified size onto map at specified row, column, and orientation
        """
        rv = 0
        cv = 0
        if orientation == Orientation.VERTICAL:
            rv = 1
        if orientation == Orientation.HORIZONTAL:
            cv = 1
        idx = size-1
        if 0 <= row < self.shots.shape[0] and 0 <= col < self.shots.shape[1] and 0 <= row + rv*idx < self.shots.shape[0] and 0 <= col + cv*idx < self.shots.shape[1]:
            if self.ships[row:row+rv*idx,col:col+cv*idx].sum() == 0:
                self.ships[row:row+rv*idx+1,col:col+cv*idx+1] = len(self.max_hits_per_ship)+1
                self.max_hits_per_ship.append(size)
                self.hits_per_ship.append(0)
                self.active_ships.append(True)
                return True
        return False

    def activeShips(self):
        """
        returns number of active ships left
        """
        return self.active_ships

    def placeShips(self, ship_sizes):
        for size in ship_sizes:
            row = randint(0,self.shots_size-1)
            col = randint(0,self.shots_size-1)
            orientation = choice(list(Orientation))
            while not self.placeShip(row, col, orientation, size):
                row = randint(0,self.shots_size-1)
                col = randint(0,self.shots_size-1)
                orientation = choice(list(Orientation))

    def reset(self, random_seed=None):
        seed(random_seed)
        self.placeShips(self.max_hits_per_ship)

def main():
    """
    Main function when script is run
    """
    b = BattleshipBoard(10)
    b.placeShips([5,4,3,3,2])
    b.fire(5, 5)
    b.fire(3, 3)
    b.printShips()
    b.printShots()


if __name__ == "__main__":
    main()

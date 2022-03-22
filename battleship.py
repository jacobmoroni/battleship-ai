"""
Battleship game class
"""
from enum import Enum
import numpy as np
from random import randint


class Orientation(Enum):
    """
    Orientation enum for directionality when placing ships
    """
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class BattleshipBoard:
    """
    BattleshipBoard class
    """

    def __init__(self, grid_size, ships):
        self.grid_size = grid_size
        self.grid = np.zeros(shape=(grid_size, grid_size))
        self.ships = []
        self.ships_active = []
        for ship in ships:
            self.placeShip(ship[0], ship[1], ship[2], ship[3])

    def fire(self, row, col):
        """
        fire at the desired row and column
        Updates the board to reflect hit or not
        Returns:
            True if you have not fired on that spot already, else False
        """
        hit = False
        if self.grid[row, col] == 0:
            hit = self.isHit(row, col)
            if hit:
                self.grid[row, col] = 1
            else:
                self.grid[row, col] = -1
            return True, hit
        print("You've already fired at ({row},{col})")
        return False, hit

    def isHit(self, row, col):
        """
        Returns True if specified row and col is a hit
        """
        for idx in range(len(self.ships)):
            if (row, col) in self.ships[idx]:
                self.ships[idx].remove((row, col))
                if len(ship) > 0:
                    print("You hit my battleship")
                else:
                    self.active_ships[idx] = False
                    print("You sunk my battleship!")
                return True
        print("You missed!")
        return False

    def printBoard(self):
        """
        Prints the current state of the board
        """
        print(self.grid)

    def placeShip(self, row, col, orientation, size):
        """
        Places ship of specified size onto map at specified row, column, and orientation
        """
        rv = 0
        cv = 0
        if orientation == Orientation.NORTH:
            rv = 1
        if orientation == Orientation.EAST:
            cv = 1
        if orientation == Orientation.SOUTH:
            rv = -1
        if orientation == Orientation.WEST:
            cv = -1
        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1] and 0 <= row + rv*size < self.grid.shape[0] and 0 <= col + cv*size < self.grid.shape[1]:
            ship = [(row+rv*idx, col+cv*idx) for idx in range(size)]
            for s in self.ships:
                for s1 in ship:
                    if s1 in s:
                        print(f"Can't place a ship at {row},{col} facing {orientation}")
                        return False
            self.ships.append(ship)
            self.ships_active.append(True)
            return True
        print("Can't place a ship here")
        return False

# return the list of ships still active
    def activeShips(self):
        """
        returns number of active ships left
        """
        return self.active_ships

    def printShips(self):
        """
        Prints grid of ship placement
        """
        ship_board = np.zeros(shape=self.grid.shape)
        for ship in self.ships:
            for point in ship:
                ship_board[point[0], point[1]] = 1
        print(ship_board)
    
    def placeShips(self, ship_sizes):
        for size in ship_sizes:
            row = randint(0,self.grid_size)
            col = randint(0,self.grid_size)
            orientation = randint(0,4)
            while not self.placeShip(row, col, orientation, size):
                row = randint(0,self.grid_size)
                col = randint(0,self.grid_size)
                orientation = randint(0,4)

def main():
    """
    Main function when script is run
    """
    b = BattleshipBoard(10, [])
    b.placeShip(5, 5, Orientation.NORTH, 2)
    b.fire(5, 5)
    b.fire(4, 5)
    b.printBoard()


if __name__ == "__main__":
    main()

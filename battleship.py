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
        for ship in ships:
            self.placeShip(ship[0], ship[1], ship[2], ship[3])

    def fire(self, row, col):
        """
        fire at the desired row and column
        Updates the board to reflect hit or not
        Returns:
            True if you have not fired on that spot already, else False
        """
        if self.grid[row, col] == 0:
            if self.isHit(row, col):
                self.grid[row, col] = 1
            else:
                self.grid[row, col] = -1
            return True
        print(f"You've already fired at ({row},{col})")
        return False

    def isHit(self, row, col):
        """
        Returns True if specified row and col is a hit
        """
        for ship in self.ships:
            if (row, col) in ship:
                ship.remove((row, col))
                if len(ship) > 0:
                    print("You hit my battleship")
                else:
                    self.ships.remove(ship)
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
            return True
        print("Can't place a ship here")
        return False

    def activeShips(self):
        """
        returns number of active ships left
        """
        return len(self.ships)

    def printShips(self):
        """
        Prints grid of ship placement
        """
        ship_board = np.zeros(shape=self.grid.shape)
        for ship in self.ships:
            for point in ship:
                ship_board[point[0], point[1]] = 1
        print(ship_board)


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

from battleship import BattleshipBoard
import numpy as np

def calcheatmap(grid,  ships):
    heatmap = np.zeros(shape=grid.shape)
    for idx in range(heatmap.shape[0]):
        for jdx in range(heatmap.shape[1]):
            for ship in ships:
                heatmap[idx,jdx] += probOfSquare(ship, grid, idx, jdx)
                if grid[idx,jdx] < 2:
                    heatmap[idx,jdx] = 0
    return heatmap

def probOfSquare(ship, grid, idx, jdx):
    val = 0
    for kdx in range(max(0,idx-ship+1), min(10-ship+1, idx+1)):
        if (0 < grid[kdx:kdx+ship,jdx]).all():
            #print(grid[kdx:kdx+ship,jdx].sum())
            val += (2-grid[kdx:kdx+ship,jdx]).sum() + 1
    for kdx in range(max(0,jdx-ship+1), min(10-ship+1, jdx+1)):
        if ( 0 < grid[idx,kdx:kdx+ship]).all():
            val += (2-grid[idx,kdx:kdx+ship]).sum() + 1
    return val

def getSunkShipPossibility(grid, ships, row, col):
    options = []
    #vertical possibility
    if (row+1 < grid.shape[0] and grid[row+1, col] == 1) or (row > 0 and grid[row-1, col] == 1):
        idx=0
        jdx=0
        while row+idx+1 < grid.shape[0] and 0 < grid[row + idx+1, col] < 2:
            idx += 1
        while row - jdx > 0 and 0 < grid[row - jdx-1, col] < 2:
            jdx += 1
        size = idx+jdx+1
        for ship in ships:
            if ship <= size: options.append(ship)
    #horizontal possibility
    if (col+1 < grid.shape[1] and grid[row, col+1] == 1) or (col > 0 and grid[row, col-1] == 1):
        idx = 0
        jdx = 0
        while col+idx+1 < grid.shape[1] and 0 < grid[row,col+idx+1] < 2:
            idx += 1
        while col-jdx > 0 and 0 < grid[row,col-jdx-1] < 2:
            jdx += 1
        size = idx + jdx + 1
        for ship in ships:
            if ship <= size: options.append(ship)
    return options

def updateSunk(grid, row, col, prob, ships):
    one = False
    right  = (col + 1 < grid.shape[1] and grid[row,col+1] == 1)
    left = (col > 0 and grid[row,col-1] == 1)
    down  = (row + 1 < grid.shape[0] and grid[row+1,col] == 1)
    up = (row > 0 and grid[row-1,col] == 1)
    minsize = min(ships)
    sizes_ordered = np.unique(ships)
    prob_ordered = np.zeros(sizes_ordered.shape[0])
    for idx in range(len(ships)):
        prob_ordered[np.where(sizes_ordered==ships[idx])] += prob[idx]
    #print(prob_ordered)
    if (up or down) and not (left or right):
        if up and down:#only vertical but both directions
            pass
        else:
            v = 1 if down else -1
            last = 0
            for jdx in range(sizes_ordered.shape[0]):
                for idx in range(last,sizes_ordered[jdx]):
                    if 0 <= row+sizes_ordered[jdx]*v < grid.shape[0]:
                        if 0 <= grid[row+idx*v,col] < 2:
                            grid[row+idx*v,col] = 1 - prob_ordered[jdx:].sum()
                    last = sizes_ordered[jdx]
            #for idx in range(minsize):
            #    grid[row+idx*v,col] = 1 + prob.sum()
    elif (left or right) and not (up or down):
        if left and right:
            pass
        else:
            v = 1 if right else -1
            last = 0
            for jdx in range(sizes_ordered.shape[0]):
                for idx in range(last,sizes_ordered[jdx]):
                    if 0 <= col+sizes_ordered[jdx]*v < grid.shape[1]:
                        if 0 <= grid[row,col+idx*v] < 2:
                            grid[row, col+idx*v] = 1 - prob_ordered[jdx:].sum()
                            #print(row, col+idx*v, grid[row,col+idx*v],1-prob_ordered[jdx:].sum(),prob_ordered[jdx:].sum())
                    last = sizes_ordered[jdx]

    else:#could be horizontal or vertical
        pass
def updateProbs(probs):
    #print('probupdate')
    for idx in reversed(range(len(probs))):
        if (probs[idx]==1).any():
            kdx = np.where(probs[idx]==1)[0][0]
            for jdx in range(len(probs)): 
                if idx != jdx: 
                    probs[jdx][kdx] = 0
                    probs[jdx] = probs[jdx]/probs[jdx].sum()

def guessProbs(probs):
    guess = np.array(probs)
    guesses = []
    normal= np.divide(guess, guess.sum(axis=0), out=np.zeros_like(guess), where=guess.sum(axis=0)!=0)
    for idx in range(normal.shape[0]):
        guesses.append(np.where(normal[idx,:]==max(normal[idx,:])))
    

def updateGrid(grid, guess, hit, sunk, prob):
    if not hit:
        grid[guess[0],guess[1]] = -1
        r1 = max(0,guess[0] -3)
        r2 = min(grid.shape[0]-1,guess[0] + 3)
        c1 = max(0,guess[1]-3)
        c2 = min(grid.shape[1]-1,guess[1]+3)
        #print(grid[r1:r2,c1:c2])
    if hit:
        grid[guess[0],guess[1]] = 1
    if sunk:
        ships = [5,4,3,3,2]
        options = getSunkShipPossibility(grid, ships, guess[0], guess[1])
        prob = np.zeros_like(ships)
        for option in options:
            for idx in range(len(ships)):
                if ships[idx] == option:
                    prob[idx] += 1
        prob = prob / prob.sum()
        #print(prob)
        updateSunk(grid,guess[0], guess[1], prob, ships)
        #print(grid)

def runMap(seed=None, boardsize=10, ships=[5,4,3,3,2]):
    np.set_printoptions(precision=2)
    b = BattleshipBoard(boardsize,seed)
    grida = np.ones(shape=(boardsize,boardsize))*2
    probs = []
    sunk_prob = np.zeros_like(ships)
    count = 0
    last_active = np.array(b.activeShips())
    while b.activeShips().sum() > 0:
        grid = b.getShots()
        active = b.activeShips()
        active_ships = [ships[idx] for idx in range(len(ships)) if active[idx]]
        heat=calcheatmap(grida,active_ships)
        guess=np.unravel_index(np.argmax(heat,axis=None),heat.shape)
        count += 1
        valid, hit, sunk = b.fire(guess[0],guess[1])
        prob = np.array(b.activeShips())^last_active
        last_active = np.array(b.activeShips())
        updateGrid(grida, guess, hit, sunk, prob)
    print(count)
    return count

import matplotlib.pyplot as plt
from time import sleep
def displayMap(seed=None, boardsize=10, ships=[5,4,3,3,2]):
    np.set_printoptions(precision=2)
    b = BattleshipBoard(boardsize,seed)
    grida = np.ones(shape=(boardsize,boardsize))*2
    probs = []
    sunk_prob = np.zeros_like(ships)
    count = 0
    last_active = np.array(b.activeShips())
    plt.ion()
    plt.show()
    while b.activeShips().sum() > 0:
        print(count)
        grid = b.getShots()
        active = b.activeShips()
        active_ships = [ships[idx] for idx in range(len(ships)) if active[idx]]
        heat=calcheatmap(grida,active_ships)
        guess=np.unravel_index(np.argmax(heat,axis=None),heat.shape)
        count += 1
        valid, hit, sunk = b.fire(guess[0],guess[1])
        prob = np.array(b.activeShips())^last_active
        last_active = np.array(b.activeShips())
        updateGrid(grida, guess, hit, sunk, prob)
        plt.imshow(heat, cmap='hot', interpolation='nearest')
        #plt.show(block=False)
        plt.pause(.1)
        #sleep(.5)
    return count


if  __name__ == "__main__":
    #b = BattleshipBoard(10,10)
    #b.printShips()
    #grid = np.ones_like(b.getShots())*2
    #guesses = [(1,3),(1,4),(1,5),(1,2)]
    #last_active = np.array(b.activeShips())
    #for guess in guesses:
    #    valid, hit, sunk = b.fire(guess[0],guess[1])
    #    prob = np.array(b.activeShips())^last_active
    #    last_active = np.array(b.activeShips())
    #    updateGrid(grid, guess, hit, sunk, prob)
    #print(grid) 
    
    displayMap(seed=10)

    #counts = []
    #for idx in range(500):
    #    counts.append(runMap())
    #print("Avg score: ", np.average(counts))

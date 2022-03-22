"""
curses practice
"""
import curses
import time
import numpy as np


def reportProgress(array, progress):
    """progress: 0-10"""
    stdscr.addstr(0, 0, str(array))
    stdscr.addstr(11, 0, f"Total progress: [{'#' * progress:10}] {progress}%")
    stdscr.refresh()


if __name__ == "__main__":
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    try:
        for i in range(10):
            reportProgress(np.ones((10, 10))*i, i+1)
            time.sleep(0.5)
    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()

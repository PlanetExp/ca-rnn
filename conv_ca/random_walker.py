"""Random walker 2D grid with lookahead, noise and negative sampling

Created: 8 May 2017
Author: Frederick Heidrich
fredheidrich@icloud.com
github.com/fredheidrich
"""
from timeit import default_timer as timer
import os
import sys
import random
# from pprint import pprint
# import time

import numpy as np
import h5py
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle

ROWS = 20
COLS = 20


def get_neighbors(frontier, grid, visited, similar_cells):
    """Generic neighbor function"""
    assert isinstance(visited, set), "param visited in not of type set"
    assert isinstance(similar_cells, set), "param similar_sets in not \
        of type set"

    if isinstance(frontier, tuple):
        frontier = set([frontier])

    neighbors = set()
    for row, col in frontier:
        for rowdir, coldir in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            rowneigh, colneigh = row + rowdir, col + coldir
            if (0 <= colneigh < grid.shape[0] and
                    0 <= rowneigh < grid.shape[1] and
                    grid[rowneigh, colneigh] not in similar_cells and
                    (rowneigh, colneigh) not in visited):
                neighbors.add((rowneigh, colneigh))
    return neighbors


def probability(prob):
    """Probability p of being true in {0-1}"""
    return random.random() <= prob


def random_walker_generator(shape, negative=False):
    """Generates a random walker on a grid of shape [width, height]
    Args:
        shape: -
        negative: True to return a negative sample
    Returns: random walker generator that yields a grid of [width, height]
    """
    attempts = 0
    while True:
        steps = 0
        found_goal = False
        grid = np.zeros(shape)
        # start on bottom row
        current = (ROWS - 1, random.randint(0, COLS - 1))
        grid[current] = 1
        steps += 1
        visited = set(current)

        neighbors = get_neighbors(current, grid, visited, similar_cells={1})
        while len(neighbors) > 0:
            for (neigh_x, neigh_y) in set(neighbors):
                # lookahead for neighbors neighbors
                lookahead = get_neighbors(
                    (neigh_x, neigh_y), grid, visited, similar_cells={1})
                if len(lookahead) < 3:  # contains neighbors with 1's
                    # edge cases
                    if neigh_x == 0 and random.random() >= 0.25:
                        # chance of reaching goal at top
                        continue
                    elif ((neigh_y == 0 or neigh_y == ROWS - 1) and
                          len(lookahead) == 2):
                        continue
                    else:
                        neighbors.remove((neigh_x, neigh_y))

            if len(neighbors) == 0:
                # print ("no more neighbors to pick")
                break

            # time.sleep(0.15)
            # os.system("clear")
            # draw_grid(grid)

            current = random.sample(neighbors, 1)[0]  # pick a random neighbor
            # print ("selected: ", current)
            grid[current] = 1
            steps += 1
            visited.add(current)
            if current[0] == 0:  # top row
                # print ("top row reached")
                found_goal = True
                break
            neighbors = get_neighbors(current, grid, visited, similar_cells={1})

        if (found_goal and not negative) or (not found_goal and negative):
            # print ("Succeeded after %d attempts" % attempts)
            attempts = 0
            grid = apply_noise(grid)
            yield grid, steps
        else:
            attempts += 1


def check_connections_length(grid):
    """Returns the longest connection from top to bottom using DFS"""
    start = [(0, n) for n in range(COLS) if grid[0, n] == 2]
    target = [(ROWS - 1, n) for n in range(COLS) if grid[ROWS - 1, n] == 2]
    generation = 0  # generation index
    visited = set()
    frontier = set(start)

    while not frontier.intersection(target):
        generation += 1
        visited.update(frontier)
        frontier = get_neighbors(frontier, grid, visited, similar_cells={0})
        # time.sleep(1)
        # pprint(frontier)
        if not frontier:  # cul de sac!
            return None
    return generation


def sever_connections(grid):
    """Helper function to make sure we don't create an accidental path
    through noise generator"""
    # print ("found accidental connection, severing:")
    rand = random.randint(0, 1) * (ROWS - 1)  # random top or bottom
    for col in range(COLS):
        if grid[rand, col] == 2:
            grid[rand, col] = 0

    # add back the safe ones
    for col in range(COLS):
        if len(get_neighbors(
                (rand, col), grid, visited=set(), similar_cells={1, 2})) == 3:
            grid[rand, col] = 2

    # draw_grid(grid)
    # sys.stdout.write("=" * COLS * 3 + "\n")
    return grid


def render_grid(grid):
    """Helper function to turn all cells to the same value"""
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row, col] != 1 and grid[row, col] != 0:
                grid[row, col] = 1
    return grid


def apply_noise(grid, prob=0.5):
    """Applies a random noise over each cell in grid to a probability
    Avoids noise near a path with lookahead"""
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row, col] == 0:
                # p = random.random() <= probability
                cell = 0
                if random.random() <= prob:
                    cell = 2
                # Make sure we only apply noise outside the path
                neighbors = get_neighbors(
                    (row, col), grid, visited=set(), similar_cells={1})
                if len(neighbors) == 4:
                    grid[row, col] = cell
                elif (len(neighbors) == 3 and
                      (row == 0 or row == ROWS - 1 or
                       col == 0 or col == COLS - 1)):
                    # Handle edge cases
                    grid[row, col] = cell
                elif (len(neighbors) == 2 and
                        ((row == 0 and col == 0) or
                            (row == 0 and col == COLS - 1) or
                            (row == ROWS - 1 and col == 0) or
                            (row == ROWS - 1 and col == COLS - 1))):
                    # Handle corner cases
                    grid[row, col] = cell

    # Could be written better, but it's still pretty fast
    if check_connections_length(grid):
        grid = sever_connections(grid)
    return render_grid(grid)


def draw_grid(grid):
    """Draws a grid of dims [width, height] to stdout"""
    for row in range(ROWS):
        for col in range(COLS):
            if grid[row, col] == 0:  # empty
                sys.stdout.write(" . ")
            elif grid[row, col] == 1:  # path
                sys.stdout.write(" X ")
            elif grid[row, col] == 2:
                sys.stdout.write(" O ")
            else:
                sys.stdout.write(" @ ")

            if col % COLS == COLS - 1:
                sys.stdout.write("\n")


def load_hdf5(filename):
    """Load a hdf5 file"""
    with h5py.File(filename, "r") as h5file:
        grids = h5file["grids"]
        steps = h5file["steps"]
        connections = h5file["connection"]

        # print (grids.shape, connections.shape)

        return grids[:], connections[:], steps[:]


def save_hdf5(grids, steps, connection, filename):
    """Saves a hdf5 file to disk"""
    mdir = os.path.dirname(filename)
    if not os.path.exists(mdir):
        os.makedirs(mdir)

    with h5py.File(filename, "w") as h5file:
        dset_grids = h5file.create_dataset(
            "grids", data=grids, dtype="i", compression="gzip")
        dset_grids.attrs["Description"] = "Grids"
        h5file["grids"].dims[0].label = "w"
        h5file["grids"].dims[1].label = "h"
        dset_steps = h5file.create_dataset(
            "steps", data=steps, dtype="i", compression="gzip")
        dset_steps.attrs["Description"] = "Number of steps"
        dset_connection = h5file.create_dataset(
            "connection", data=connection, dtype="i", compression="gzip")
        dset_connection.attrs["Description"] = "Connected or not"


def main():
    """Creates a grid with a random walker on it"""
    start = timer()
    num_pos = 50000
    num_neg = 50000
    tot_examples = num_pos + num_neg
    grids = np.empty((tot_examples, ROWS, COLS))
    steps = np.empty((tot_examples))
    connection = np.empty((tot_examples))
    generate_positive_grid = random_walker_generator(
        (ROWS, COLS), negative=False)
    generate_negative_grid = random_walker_generator(
        (ROWS, COLS), negative=True)

    for i in range(num_pos):
        grids[i], steps[i] = next(generate_positive_grid)
        connection[i] = 1

    for i in range(num_neg):
        i = i + num_pos
        grids[i], steps[i] = next(generate_negative_grid)
        connection[i] = 0

    end = timer()

    # plot_histogram(steps[:num_pos], title="Positive examples distribution")
    # plot_histogram(steps[num_pos:], title="Negative examples distribution")

    grids, steps, connection = shuffle(
        grids, steps, connection, random_state=1234)

    # print sample to terminal
    sample = random.randint(0, tot_examples - 1)
    draw_grid(grids[sample])
    print ("sample no. %d: %d steps. Connection: %d" %
           (sample, steps[sample], connection[sample]))
    print ("Time: %.3fs" % (end - start))

    filename = "tmp/connectivity.h5"
    save_hdf5(grids, steps, connection, filename)


if __name__ == "__main__":
    # main()
    load_hdf5("tmp/connectivity.h5")

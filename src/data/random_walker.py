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
# import argparse
# from pprint import pprint
# import time

import numpy as np
import h5py
from sklearn.utils import shuffle


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
            if (0 <= colneigh < grid.shape[1] and
                    0 <= rowneigh < grid.shape[0] and
                    grid[rowneigh, colneigh] not in similar_cells and
                    (rowneigh, colneigh) not in visited):
                neighbors.add((rowneigh, colneigh))
    return neighbors


def probability(prob):
    """Probability p of being true in {0-1}"""
    return random.random() <= prob


def random_walker_generator(rows, cols, negative=False):
    """Generates a random walker on a grid of shape [rows, cols]
    Args:
        shape: -
        negative: True to return a negative sample
    Returns: random walker generator that yields a grid of [rows, cols]
    """
    attempts = 0
    while True:
        steps = 0
        found_goal = False
        grid = np.zeros((rows, cols))
        # start on bottom row
        current = (rows - 1, random.randint(0, cols - 1))
        grid[current] = 1
        steps += 1
        visited = set(current)

        connection = 0

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
                    elif ((neigh_y == 0 or neigh_y == rows - 1) and
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

            # hack
            # override above step counter, because the random noise
            # might have added more, shorter connections
            # we do this because network was picking up patterns
            # from making random noise not entirely random
            steps, connected = check_connections_length(grid)
            if connected and negative:
                    continue

            # randomly flip grid upside down
            if random.random() <= 0.5:
                grid = np.flipud(grid)

            yield grid, steps, connected
        else:
            attempts += 1


def check_connections_length(grid):
    rows = grid.shape[0]
    cols = grid.shape[1]
    """Returns the longest connection from top to bottom using DFS"""
    top = [(0, c) for c in range(cols) if grid[0, c] == 1]
    bottom = [(rows - 1, c) for c in range(cols) if grid[rows - 1, c] == 1]
    generation = 0  # generation index
    visited = set()
    frontier = set(bottom)

    while not frontier.intersection(top):
        generation += 1
        visited.update(frontier)
        frontier = get_neighbors(frontier, grid, visited, similar_cells={0})
        # time.sleep(1)
        if not frontier:  # cul de sac!
            return generation, False
            # return None
    return generation, True


def sever_connections(grid):
    rows = grid.shape[0]
    cols = grid.shape[1]
    """Helper function to make sure we don't create an accidental path
    through noise generator"""
    # print ("found accidental connection, severing:")
    rand = random.randint(0, 1) * (rows - 1)  # random top or bottom
    for col in range(cols):
        if grid[rand, col] == 2:
            grid[rand, col] = 0

    # add back the safe ones
    for col in range(cols):
        if len(get_neighbors(
                (rand, col), grid, visited=set(), similar_cells={1, 2})) == 3:
            grid[rand, col] = 2

    # draw_grid(grid)
    # sys.stdout.write("=" * cols * 3 + "\n")
    return grid


def render_grid(grid):
    """Helper function to turn all cells to the same value"""
    rows = grid.shape[0]
    cols = grid.shape[1]
    for row in range(rows):
        for col in range(cols):
            if grid[row, col] != 1 and grid[row, col] != 0:
                grid[row, col] = 1
    return grid


def apply_noise(grid, prob=0.2):
    """Applies a random noise over each cell in grid to a probability
    Avoids noise near a path with lookahead"""
    rows = grid.shape[0]
    cols = grid.shape[1]
    for row in range(rows):
        for col in range(cols):
            if grid[row, col] == 0:
                # p = random.random() <= probability
                cell = 0
                if random.random() <= prob:
                    cell = 2
                # Make sure we only apply noise outside the path
                # neighbors = get_neighbors(
                #     (row, col), grid, visited=set(), similar_cells={1})
                # if len(neighbors) == 4:
                #     grid[row, col] = cell
                # elif (len(neighbors) == 3 and
                #       (row == 0 or row == rows - 1 or
                #        col == 0 or col == cols - 1)):
                #     # Handle edge cases
                #     grid[row, col] = cell
                # elif (len(neighbors) == 2 and
                #         ((row == 0 and col == 0) or
                #             (row == 0 and col == cols - 1) or
                #             (row == rows - 1 and col == 0) or
                #             (row == rows - 1 and col == cols - 1))):
                #     # Handle corner cases
                    grid[row, col] = cell

    # Could be written better, but it's still pretty fast
    # if check_connections_length(grid):
    #     grid = sever_connections(grid)
    return render_grid(grid)


def draw_grid(grid):
    """Draws a grid of dims [width, height] to stdout"""
    rows = grid.shape[0]
    cols = grid.shape[1]
    for row in range(rows):
        for col in range(cols):
            if grid[row, col] == 0:  # empty
                sys.stdout.write(" . ")
            elif grid[row, col] == 1:  # path
                sys.stdout.write(" X ")
            elif grid[row, col] == 2:
                sys.stdout.write(" O ")
            else:
                sys.stdout.write(" @ ")

            if col % cols == cols - 1:
                sys.stdout.write("\n")


def check_repeats(grids):
    rows = grids.shape[1]
    cols = grids.shape[2]
    """Check if the dataset is repeating or not by converting each image
    to a hashed byte-array, then iterating though the new array and look for
    duplicates. Each hash should be unique."""
    from collections import Counter
    a = [hash(item.data.tobytes()) for item in grids.reshape(-1, rows * cols)]
    # pprint(a)
    # print(len(a))
    b = [item for item, count in Counter(a).items() if count > 1]
    assert len(b) == 0, "Dataset contains repeats"
    print ("No repeats found out of %d samples." % grids.shape[0])


def create_random_walkers(rows, cols, num_examples, positive_fraction=0.5):
    """Creates a grid with a random walker on it
    Returns:
        grids, steps, connection
    """
    start = timer()
    num_pos = int(num_examples * positive_fraction)
    num_neg = num_examples - num_pos

    assert num_pos + num_neg == num_examples

    grids = np.empty((num_examples, rows, cols))
    steps = np.empty((num_examples))
    connection = np.empty((num_examples))
    pos_grid_generator = random_walker_generator(
        rows, cols, negative=False)
    neg_grid_generator = random_walker_generator(
        rows, cols, negative=True)

    for i in range(num_pos):
        grids[i], steps[i], connection[i] = next(pos_grid_generator)
        # connection[i] = 1

    for i in range(num_neg):
        i = i + num_pos
        grids[i], steps[i], connection[i] = next(neg_grid_generator)
        # connection[i] = 0

    end = timer()

    # plot_histogram(steps[:num_pos], title="Positive examples distribution")
    # plot_histogram(steps[num_pos:], title="Negative examples distribution")

    grids, steps, connection = shuffle(
        grids, steps, connection)

    draw_random_sample(grids, steps, connection)
    print ("Time: %.3fs" % (end - start))

    return grids, steps, connection


def draw_random_sample(grids, steps=None, connection=None):
    """print random sample to terminal"""
    sample = random.randint(0, grids.shape[0] - 1)
    draw_grid(grids[sample])
    print ("sample no. %d" % sample)
    if steps is not None:
        print ("steps: %d" % steps[sample])
    if connection is not None:
        print ("connection: %d" % connection[sample])


def test():
    grids, steps, connection = create_random_walkers(14, 14, 10)


if __name__ == "__main__":
    test()
    # grids, _, _ = load_hdf5("data/connectivity_8x8.h5")
    # check_repeats(grids)

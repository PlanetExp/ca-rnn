"""Random walker 2D grid with lookahead, noise and negative sampling

Created: 8 May 2017
Author: Frederick Heidrich
fredheidrich@icloud.com
github.com/fredheidrich
"""
import numpy as np
from pprint import pprint
import random
import time
import os
import sys
import h5py
import seaborn as sns
import pandas as pd
from timeit import default_timer as timer
from sklearn.utils import shuffle

rows = 20
cols = 20


def get_neighbors(frontier, grid, visited=set(), ks={1}):
    if isinstance(frontier, tuple):
        frontier = set([frontier])

    neighbors = set()
    for r, c in frontier:
        for rd, cd in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            rn, cn = r + rd, c + cd
            if (0 <= cn < grid.shape[0] and 0 <= rn < grid.shape[1] and
                    grid[rn, cn] not in ks and (rn, cn) not in visited):
                neighbors.add((rn, cn))
    return neighbors


def probability(p):
    """Probability p of being true in {0-1}"""
    return random.random() <= p


def random_walker_generator(shape, negative=False):
    attempts = 0
    while True:
        steps = 0
        found_goal = False
        grid = np.zeros(shape)
        # start on bottom row
        current = (rows - 1, random.randint(0, cols - 1))
        grid[current] = 1
        steps += 1
        visited = set(current)

        N = get_neighbors(current, grid, visited)
        while len(N) > 0:
            for (xn, yn) in set(N):
                NN = get_neighbors((xn, yn), grid, visited)
                if len(NN) < 3:  # contains neighbors with 1's
                    # edge cases
                    if xn == 0 and random.random() >= 0.25:
                        continue  # 50/50 chance of reaching goal at top
                    elif (yn == 0 or yn == rows - 1) and len(NN) == 2:
                        continue
                    else:
                        N.remove((xn, yn))

            if len(N) == 0:
                # print ("no more neighbors to pick")
                break

            # time.sleep(0.15)
            # os.system("clear")
            # draw_grid(grid)

            current = random.sample(N, 1)[0]  # pick a random neighbor
            # print ("selected: ", current)
            grid[current] = 1
            steps += 1
            visited.add(current)
            if current[0] == 0:  # top row
                # print ("top row reached")
                found_goal = True
                break
            N = get_neighbors(current, grid, visited)

        if (found_goal and not negative) or (not found_goal and negative):
            # print ("Succeeded after %d attempts" % attempts)
            attempts = 0
            grid = apply_noise(grid)
            yield grid, steps
        else:
            attempts += 1


def check_connections_length(grid):
    start = [(0, n) for n in range(cols) if grid[0, n] == 2]
    target = [(rows - 1, n) for n in range(cols) if grid[rows - 1, n] == 2]
    g = 0  # generation index
    visited = set()
    frontier = set(start)

    while not frontier.intersection(target):
        g += 1
        visited.update(frontier)
        frontier = get_neighbors(frontier, grid, visited, ks={0})
        # time.sleep(1)
        # pprint(frontier)
        if not frontier:  # cul de sac!
            return None
    return g


def sever_connections(grid):
    # make sure we don't accidentally create a path with noise
    # print ("found accidental connection, severing:")
    r = random.randint(0, 1) * (rows - 1)  # random top or bottom
    for n in range(cols):
        if grid[r, n] == 2:
            grid[r, n] = 0

    # add back the safe ones
    for n in range(cols):
        if len(get_neighbors((r, n), grid, ks={1, 2})) == 3:
            grid[r, n] = 2

    # draw_grid(grid)
    # sys.stdout.write("=" * cols * 3 + "\n")
    return grid


def render_grid(grid):
    for m in range(rows):
        for n in range(cols):
            if grid[m, n] != 1 and grid[m, n] != 0:
                grid[m, n] = 1
    return grid


def apply_noise(grid, probability=0.5):
    # with lookahead
    for m in range(rows):
        for n in range(cols):
            if grid[m, n] == 0:
                # p = random.random() <= probability
                p = 0
                if random.random() <= probability:
                    p = 2
                # Make sure we only apply noise outside the path
                N = get_neighbors((m, n), grid)
                if len(N) == 4:
                    grid[m, n] = p
                elif (len(N) == 3 and
                        (m == 0 or m == rows - 1 or n == 0 or n == cols - 1)):
                    # Handle edge cases
                    grid[m, n] = p
                elif (len(N) == 2 and
                        ((m == 0 and n == 0) or (m == 0 and n == cols - 1) or
                            (m == rows - 1 and n == 0) or
                            (m == rows - 1 and n == cols - 1))):
                    # Handle corner cases
                    grid[m, n] = p

    if check_connections_length(grid):
        grid = sever_connections(grid)
    return render_grid(grid)


def draw_grid(grid):
    for m in range(rows):
        for n in range(cols):
            if grid[m, n] == 0:  # empty
                sys.stdout.write(" . ")
            elif grid[m, n] == 1:  # path
                sys.stdout.write(" X ")
            elif grid[m, n] == 2:
                sys.stdout.write(" O ")
            else:
                sys.stdout.write(" @ ")

            if n % cols == cols - 1:
                sys.stdout.write("\n")


def load_hdf5(filename):
    with h5py.File(filename, "r") as f:
        grids = f["grids"]
        steps = f["steps"]
        connections = f["connection"]

        print (grids.shape, connections.shape)
        # plot_histogram(steps[connections[:] == 1], "Positive sample distribution")
        # plot_histogram(steps[connections[:] == 0], "Negative sample distribution")


def save_hdf5(grids, steps, connection, filename):
    mdir = os.path.dirname(filename)
    if not os.path.exists(mdir):
        os.makedirs(mdir)

    with h5py.File(filename, "w") as f:
        dset_grids = f.create_dataset(
            "grids", data=grids, dtype="i", compression="gzip")
        dset_grids.attrs["Description"] = "Grids"
        f["grids"].dims[0].label = "w"
        f["grids"].dims[1].label = "h"
        dset_steps = f.create_dataset(
            "steps", data=steps, dtype="i", compression="gzip")
        dset_steps.attrs["Description"] = "Number of steps"
        dset_connection = f.create_dataset(
            "connection", data=connection, dtype="i", compression="gzip")
        dset_connection.attrs["Description"] = "Connected or not"


def main():
    start = timer()
    num_pos = 50000
    num_neg = 50000
    tot_examples = num_pos + num_neg
    grids = np.empty((tot_examples, rows, cols))
    steps = np.empty((tot_examples))
    connection = np.empty((tot_examples))
    generate_positive_grid = random_walker_generator(
        (rows, cols), negative=False)
    generate_negative_grid = random_walker_generator(
        (rows, cols), negative=True)

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


def plot_histogram(steps, title):
    # plot histogram
    x = pd.Series(steps, name="connection length")
    # sns.set(rc={"figure.figsize": (8, 4)})

    sns.set_style("white")
    sns.set_style("ticks")
    ax = sns.distplot(x, hist_kws=dict(edgecolor="k", linewidth=1))
    sns.set(font="Helvetica")
    sns.despine()
    ax.set_title(title)
    sns.plt.show()


if __name__ == "__main__":
    # main()
    load_hdf5("tmp/connectivity.h5")

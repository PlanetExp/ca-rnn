import numpy as np
import random


def blobs(
        width=8,
        height=8,
        k=3,
        min_extend=0,
        max_extend=1,
        random_seed=None):
    """Generates a 1 to k number of blobs on a 2D grid

    Args:
        width: width of grid generated
        height: height of grid generated
        k: number of random points
        min_extend: min cells to recursively extend for each neighbor of k
        max_extend: max cells to recursively extend for each neighbor of k
        random_seed: not used

    Returns: grid, count: count of blobs
    """
    grid = np.zeros((width, height), dtype=np.uint8)

    def getNeighbours(coordinates, visited):
        """Return the neighbors of coordinates, that are not in visited (taboo set)
            The neighbors are represented with the pairs of their (row, column) coordinates.

            Args:
                coordinates: set of (row, column) coordinate pairs
                visited: a taboo set

            Returns:
                neighbors: list of (r, c) pairs that are neighbors to coordinates
        """
        neighbors = []
        for r, c in coordinates:
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                rn, cn = r + dr, c + dc
                if ((0 <= rn < width) and (0 <= cn < height) and grid[rn, cn] != 1 and (rn, cn) not in visited):
                    neighbors.append((rn, cn))
        return neighbors

    filled = set()

    # STEP 1: take k random points
    for i in range(k):
        rc = random.randint(0, width - 1)
        rr = random.randint(0, height - 1)
        cell = (rr, rc)
        grid[cell] = 1

        filled.add(cell)
        neighbours = getNeighbours(filled, filled)

        # STEP 2: extend recursively in min to max distance randomly
        # from every k point
        l = random.randint(min_extend, max_extend)
        while l > 0:
            l -= 1
            for i in range(len(neighbours)):
                r = random.randint(0, len(neighbours) - 1)
                neighbor = neighbours[r]
                if neighbor not in filled:
                    grid[neighbor] += 1
                    filled.add(neighbor)

            neighbours = getNeighbours(filled, filled)

    visited = set()

    def get_blob_size(i, j):
        """Add 1 for every neighbor that is in filled, else return 0"""
        if 0 < i > width or 0 < j > height or (i, j) not in filled or (i, j) in visited:
            return 0
        visited.add((i, j))
        size = 1

        size += get_blob_size(i + 1, j)
        size += get_blob_size(i - 1, j)
        size += get_blob_size(i, j + 1)
        size += get_blob_size(i, j - 1)
        return size

    # STEP 3: count blobs
    count = 0
    for r, c in filled:
        if (get_blob_size(r, c) > 0):
            count += 1

    return grid, count


def blob_generator(*args, **kwargs):
    while True:
        yield blobs(*args, **kwargs)


if __name__ == "__main__":
    b = blob_generator(14, 14, k=4, max_extend=2)
    grid, count = next(b)
    print (grid, count)

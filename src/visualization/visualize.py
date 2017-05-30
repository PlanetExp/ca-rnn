"""API for visualizing results and saving figures"""
from enum import IntEnum

from .plots import scrape_all_data
from .plots import scrape_data
from .plots import box_plot
from .plots import plot_histogram
from .plots import plot_grid
from .plots import plot

import random

from pprint import pprint 


class Plot(IntEnum):
    BOX_PLOT = 1,
    HISTOGRAM = 2,
    GRID = 3,
    PLOT = 4


def draw_plot(in_path, p, data, out_path=None, save=False):

    # data = {1: {1: [5.5, 0.5, 3.3, 0.3], 2: [10.5, 2.5, 0.1, 0.3]},
            # 2: {3: [50.5, 0.5, 5.3, 0.3], 5: [10.5, 6.5, 0.1, 0.3], 6: [88.5, 6.5, 33.1, 0.3]}}
    



    if p == Plot.BOX_PLOT:
        data = scrape_all_data(in_path)
        box_plot(data)
    elif p == Plot.HISTOGRAM:
        x = data[0]
        plot_histogram(x[:] == 1, title="Positive sample distribution")
        plot_histogram(x[:] == 0, title="Negative sample distribution")
    elif p == Plot.GRID:
        x = data[0]
        y = data[1]
        sample = random.randint(0, x.shape[0] - 1)
        title = "connectivity: " + str(y[sample])
        plot_grid(x[sample], title)
    elif p == Plot.PLOT:
        data = scrape_data(in_path)
        plot(data)





    # Training/Testing misclassification plots
    # data = scrape_data(DIR)
    # plot(data)

    # grids, connection, steps = load_hdf5("data/20x20/connectivity.h5")
    # Histograms
    # plot_histogram(
    # steps[connections[:] == 1], title="Positive sample distribution",
    #    name="connection length")
    # plot_histogram(
    # steps[connections[:] == 0], title="Negative sample distribution",
    # name="connection length")

    # Plot sample grid
    # sample = random.randint(0, grids.shape[0] - 1)
    # title = "steps=" + str(steps[sample]) + " connection=" + str(connection[sample])
    # plot_grid(grids[sample], title)


if __name__ == '__main__':
    pass

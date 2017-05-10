"""Plots various plots for data to conv_ca

Author: Frederick Heidrich
fredheidrich@icloud.com
github.com/fredheidrich
"""
import operator as op
import os
import re
import random

# import tensorflow as tf
import pandas as pd
import seaborn as sns  # pylint: disable=import-error
# import numpy as np
import matplotlib.pyplot as plt
from random_walker import load_hdf5


from tensorflow.python.summary.event_multiplexer import EventMultiplexer
from tensorflow.python.summary.event_accumulator import EventAccumulator
from pprint import pprint

RUN = "run1"
HPARAM = "layers=3,state=2"
DIR = "tmp/train/20x20/"
FULL_PATH = DIR + HPARAM


def plot_grid(data, title=None):
    fig = plt.figure(1, figsize=(5,5))
    sns.axes_style("white")
    axes = sns.heatmap(data, cbar=False, linewidths=0.2)
    axes.set_title(title)
    sns.plt.show()
    fig.savefig(title+".pdf", bbox_inches="tight")


def box_plot(data):

    for key in data.keys():
        d = data[key]
        sorted_keys, sorted_vals = zip(*sorted(d.items(), key=op.itemgetter(0)))

        fig = plt.figure(1)
        sns.set(context='notebook', style='ticks', font="Helvetica")
        # sns.set_style("ticks", {"axes.edgecolor": "0"})
        sns.utils.axlabel("state_dimension", "misclassification rate")
        ax1 = sns.boxplot(data=sorted_vals, color="white", linewidth=0.9, width=0.15)
        # sns.swarmplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.9)
        ax1.set_title("layer="+str(key))
        # category labels
        sns.plt.xticks(plt.xticks()[0], sorted_keys)
        sns.despine()
        for i, box in enumerate(ax1.artists):
            box.set_edgecolor("black")
            box.set_facecolor("white")

            for j in range(6*i,6*(i+1)):
                ax1.lines[j].set_color("black")

            ax1.lines[6 * i].set_linestyle(ls="dashed")
            ax1.lines[6 * i + 1].set_linestyle(ls="dashed")
        sns.plt.show()
        fig.savefig("layer="+ str(key) +".pdf", bbox_inches="tight")


def plot_histogram(data, title=None, name=None):
    """Plot histograms over a distribution"""
    series = pd.Series(data, name=name)
    # sns.set(rc={"figure.figsize": (8, 4)})

    sns.set_style("white")
    sns.set_style("ticks")
    axes = sns.distplot(series, hist_kws=dict(edgecolor="k", linewidth=1))
    sns.set(font="Helvetica")
    sns.despine()
    axes.set_title(title)
    sns.plt.show()


def plot(data):
    fig = plt.figure(1)
    sns.set_style("ticks")
    train, test = plt.plot(*data)
    plt.legend([train, test], ["train", "test"])
    plt.title("Misclassification rate (" + HPARAM + ")")
    sns.despine(trim=True)
    plt.ylabel("fraction of misclassification")
    plt.xlabel("step")
    plt.show()
    fig.savefig("msc"+ HPARAM +".pdf", bbox_inches="tight")


def scrape_all_data(directory):
    """Scrape a directory to find all runs and return a dictionary of the data
    using a Tensorflow multiplexer
    Args:
        directory:
    Returns:
        data: dictionary of {layer: {state: data}}
    """
    data = {}
    # Create one multiplexer for all settings that contains all sub-runs
    paths = next(os.walk(directory))
    for path in paths[1]:
        match = re.match(r"layers=(\d+),state=(\d+)", path)
        assert match, "Directory structure not 'layers=i,state=j' for %s" % path
        layer = match.group(1)
        state = match.group(2)

        # Add all runs from directory
        full_path = paths[0] + path
        multiplexer = EventMultiplexer().AddRunsFromDirectory(full_path)
        multiplexer.Reload()

        # Extract scalar summaries
        scalars = []
        for path in next(os.walk(full_path))[1]:
            scalars.extend([event.value for event in multiplexer.Scalars(
                path, "avg_accuracy/test")])

        # put in dictionary of {layer: {state: data}}
        data.setdefault(layer, {}).update({state: scalars})
    return data

def scrape_data(directory):
    full_path = os.path.join(directory, HPARAM, RUN)
    accumulator = EventAccumulator(full_path)
    accumulator.Reload()

    train_scalar = [event.value for event in accumulator.Scalars("avg_accuracy/train")]
    train_steps = [event.step for event in accumulator.Scalars("avg_accuracy/train")]
    test_scalar = [event.value for event in accumulator.Scalars("avg_accuracy/test")]
    test_steps = [event.step for event in accumulator.Scalars("avg_accuracy/test")]

    return data

def main():
    # layer/state dim accuracy box plots
    # data = {1: {1: [5.5, 0.5, 3.3, 0.3], 2: [10.5, 2.5, 0.1, 0.3]},
    #         2: {3: [50.5, 0.5, 5.3, 0.3], 5: [10.5, 6.5, 0.1, 0.3], 6: [88.5, 6.5, 33.1, 0.3]}}
    # data = scrape_all_data(DIR)
    # box_plot(data)

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
    main()

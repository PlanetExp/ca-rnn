"""API to src"""

import click

from src.data.saver import Saver
from src.data.random_walker import create_random_walkers
from src.visualization.visualize import draw_plot
from src.visualization.visualize import Plot  # intenum
from src.models.train_model import train

# tmp
from src.models.utils import generate_constrained_dataset
from src.models.dataset import load_hdf5


import h5py


@click.group()
# @click.option('-w', '--width', default=9, type=int)
# @click.argument('input_filepath', type=click.Path())
# @click.argument('output_filepath', type=click.Path())
def cli():
    pass


@click.command()
@click.option('-h', '--height', default=14, type=int)
@click.option('-w', '--width', default=14, type=int)
@click.option('-n', '--num-examples', default=60000, type=int)
def data(width, height, num_examples):

    # random walkers
    # x, _, y = create_random_walkers(width, height, num_examples)


    # random grid    
    x, y = generate_constrained_dataset((width,height), num_examples, stone_probability=0.45)

    for i, a in enumerate(y):
        if a > 1:
            y[i] = 1
        else:
            y[i] = 0

    # print ("x: %s" % x)
    # print ("y: %s" % y)
    saver = Saver("data/test.h5")
    saver.save_hdf5(x, y)


@click.command()
@click.argument('in_path', type=click.Path())
@click.argument('plot_type', type=int)
def vizualization(in_path, plot_type):

    # assert plot_type in Plot, "Plot %d not a supported type" % plot_type

    data = []
    if plot_type == Plot.HISTOGRAM or plot_type == Plot.GRID:
        x, y = load_hdf5(in_path)
        data.append(x)
        data.append(y)

        draw_plot(in_path, plot_type, data)
    else:
        draw_plot(in_path, plot_type, data)


cli.add_command(vizualization)
cli.add_command(data)


if __name__ == '__main__':
    cli()


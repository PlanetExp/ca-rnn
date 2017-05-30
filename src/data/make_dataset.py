"""API for creating data"""
# -*- coding: utf-8 -*-
import os
import logging
import click
# from dotenv import find_dotenv, load_dotenv
from random_walker import create_grids, save_hdf5
from blobs import blob_generator
from saver import Saver


import numpy as np


def generate_data(generator, num_examples, height, width, dtype=np.uint8):

    examples = np.empty((num_examples, height, width), dtype=dtype)
    labels = np.empty((num_examples), dtype=dtype)

    # examples[0], labels[0] = next(generator)
    # examples[0], labels[0] = next(generator)
    # examples[0], labels[0] = next(generator)
    # examples[0], labels[0] = next(generator)

    for i in range(num_examples):
        examples[i], labels[i] = next(generator)

    return examples, labels


def generate_name(name, num_examples, height, width):
    suffix = "_" + str(height) + "x" + str(width) + ".h5"
    return name + "_" + str(num_examples) + suffix


@click.command()
@click.option('-w', '--width', default=9, type=int)
@click.option('-h', '--height', default=9, type=int)
@click.option('-n', '--num-examples', default=10000, type=int)
@click.option('-k', '--num-points', default=4, type=int)
@click.option('-m', '--max_extend', default=1, type=int)
@click.option('-p', '--positive-fraction', default=0.5, type=float)
@click.argument('problem', type=str)
@click.argument('output_filepath', type=click.Path())
def main(height, width, num_examples, num_points, max_extend, positive_fraction, problem, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    available_problems = ['blobs', 'random-walker']
    assert problem in available_problems, "Type has to be one of: %s" % available_problems

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if problem == 'random_walker':
        grids, steps, connection = create_grids(
            height, width, num_examples, positive_fraction)
        save_hdf5(grids, steps, connection, output_filepath)
    elif problem == 'blobs':
        blobs = blob_generator(width, height, k=num_points, max_extend=max_extend)
        examples, labels = generate_data(blobs, num_examples, height, width)


        # k is number of blobs generated
        # to get class index we need to subtract 1 to get 0 to k-1 classes
        labels -= 1
    else:
        print ('no such type')

    # generate name and save file
    name = generate_name(problem, num_examples, height, width)
    path = os.path.join(output_filepath, name)
    saver = Saver(path)
    saver.save_hdf5(examples, labels)
    print ('Saved ' + path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()

    # path = "data/blobs_5000_8x8.h5"
    # a, b = load_hdf5(path)
    # # saver = Saver(path)
    # # a,b = saver.load_hdf5()
    # print (b[1:50])

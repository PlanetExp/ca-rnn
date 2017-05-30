"""Generic Dataset class wrapper"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import h5py
from collections import namedtuple
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

Datasets = namedtuple('Datasets', ['train', 'test'])


class Dataset(object):
    """Generic Dataset class wrapper"""

    def __init__(self, X, y):
        self._num_examples = X.shape[0]
        self._inputs = X
        self._labels = y
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def inputs(self):
        """Returns inputs"""
        return self._inputs

    @property
    def labels(self):
        """Returns data labels"""
        return self._labels

    @property
    def num_examples(self):
        """Returns total numbers of examples in set"""
        return self._num_examples

    @property
    def epochs_completed(self):
        """Return the float number of epochs this dataset has completed"""
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle_data=True):
        """Return the next batch_size examples from Dataset

        Args:
            batch_size

        Return:
            (inputs, labels) tuple
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle data
            if shuffle_data:
                self._inputs, self._labels = shuffle(
                    self._inputs, self._labels)

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, "batch_size exceeds \
                number of examples in Dataset"

        end = self._index_in_epoch
        return self._inputs[start:end], self._labels[start:end]


def create_datasets(inputs, labels, test_fraction=0.5):
    """Process the saved dataset and create a dataset

    Returns:
        Datasets: A named tuple collection of train and validation
            datasets objects
    """
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(
        inputs, labels, test_size=test_fraction)

    train = Dataset(inputs_train, labels_train)
    test = Dataset(inputs_test, labels_test)
    return Datasets(train=train, test=test)


def load_hdf5(filename):
    """Load a hdf5 file
    Args:
        filename: filename to load data

    Returns:
        grids, connections, steps: tuple
    """
    with h5py.File(filename, "r") as h5file:
        examples = h5file["examples"]
        labels = h5file["labels"]
        # connections = h5file["connection"]

        print (examples.shape, labels.shape)
        print ("labels[0]: %s" % labels[0])
        print ("examples[0]: %s" % examples[0])

        return examples[:], labels[:]

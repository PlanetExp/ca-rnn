"""Generic Dataset class wrapper"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# import numpy as np
from numpy.random import permutation
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

    def next_batch(self, batch_size, shuffle_data=True, random_state=None):
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
                # Lazy version below without sklearn dependency
                # perm = permutation(self._num_examples)
                # self._inputs = self._inputs[perm]
                # self._labels = self._labels[perm]
                self._inputs, self._labels = shuffle(
                    self._inputs, self._labels, random_state=random_state)

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, "batch_size exceeds \
                number of examples in Dataset"
            # while batch_size > self._num_examples:  # wrap epochs
            #     batch_size -= self._num_examples
            #     return self.batch(self._num_examples)

        end = self._index_in_epoch
        return self._inputs[start:end], self._labels[start:end]


def create_datasets(inputs, labels, test_size=0.5, random_state=1234):
    """Process the saved dataset and create a dataset

    Returns:
        Datasets: A named tuple collection of train and validation
            datasets objects
    """

    # test_split = int(len(inputs) * test_size)
    # inputs_train = inputs[:test_split]
    # labels_train = labels[:test_split]
    # inputs_test = inputs[test_split:]
    # labels_test = labels[test_split:]

    # perm = permutation(inputs_train.shape[0])
    # inputs_train = inputs_train[perm]
    # labels_train = labels_train[perm]

    # train = Dataset(inputs_train, labels_train)
    # test = Dataset(inputs_test, labels_test)
    # return Datasets(train=train, test=test)


    # below sklearn dependant
    # could implement if you got a couple hours to spare:
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_split.py
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(
        inputs, labels, test_size=test_size, random_state=random_state)

    train = Dataset(inputs_train, labels_train)
    test = Dataset(inputs_test, labels_test)
    return Datasets(train=train, test=test)

"""Test the Dataset class"""

import tensorflow as tf
import numpy as np
from pprint import pprint
import os

from dataset import create_datasets
from random_walker import load_hdf5


class DatasetTests(tf.test.TestCase):

    def testBatch(self):
        """Tests batch"""
        grids, connections, _ = load_hdf5("data/20x20/connectivity.h5")
        dsets = create_datasets(grids, connections)

        # pprint (dsets.test.inputs[10000])
        print (dsets.test.num_examples)

        i, l = dsets.train.next_batch(100)
        i2, l2 = dsets.train.next_batch(100)

        # test shuffling
        i3, l3 = dsets.test.next_batch(50000, shuffle_data=False)
        i4, l4 = dsets.test.next_batch(50000, shuffle_data=False)
        np.testing.assert_equal(i3, i4)

        print (i3[0], l3[0])
        print (i3[0], l3[0])




if __name__ == "__main__":
    tf.test.main()

"""test network slice"""

import tensorflow as tf
import numpy as np
from pprint import pprint
import os


class DatasetTests(tf.test.TestCase):

    def testNetwork(self):
        w = 5
        # create grid of zeros
        grid = np.zeros((w, w))

        # fill top row with one
        grid[:1, :w] = 1

        # fill batch
        batch = 10
        grids = np.empty((batch, w, w))
        for i in range(batch):
            grids[i] = grid

        # create tensor
        # Shape: (batch, height, width)
        a = tf.constant(grids)
        print ("a.get_shape(): %s" % a.get_shape())

        # slice tensor
        # Shape: (batch, 1, width)
        b = a[:, :1, :]
        print ("b.get_shape(): %s" % b.get_shape())


        with tf.Session():
            print ("a.eval():\n%s" % a.eval()[0:2])
            print("b.eval():\n%s" % b.eval()[0:2])

if __name__ == "__main__":
    tf.test.main()

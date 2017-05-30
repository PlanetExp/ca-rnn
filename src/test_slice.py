"""Test the Dataset class"""

import tensorflow as tf
import numpy as np
from pprint import pprint
import os


class DatasetTests(tf.test.TestCase):

    def testSlice(self):
        
        a = tf.constant(np.arange((10*4*8)), shape=[10, 4, 8])
        b = tf.slice(a, [0,0,0], [-1, 1, 8])
        with tf.Session() as sess:
            print(a.eval()[0])
            print(a.eval()[0,0,3])
            print(b.eval()[0])

if __name__ == "__main__":
    tf.test.main()

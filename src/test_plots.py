"""Test the Dataset class"""

import tensorflow as tf
import numpy as np

from pprint import pprint
import os
import random

import operator as op


class DatasetTests(tf.test.TestCase):

    def testPlots(self):

        data = {
                1: {
                    1: [1,2,3], 
                    11: [4,5,6] },
                2: {
                    1: [1,2,3], 
                    10: [7,8,9],
                    2: [4,5,6]}
                    
                }

        for key in data.keys():
            print ("key: %s" % key)


            d = data[key]
            # od = OrderedDict(sorted(d.items()))

            sorted_keys, sorted_vals = zip(*sorted(d.items(), key=op.itemgetter(1)))

            print ("sorted_keys:", sorted_keys)
            print ("sorted_vals:", sorted_vals)
            print ("type(sorted_keys):", type(sorted_keys[0]))

            a = ('1', '2')
            b = [int(x) for x in a]

            print ("a:",a)
            print ("b:", b)



if __name__ == "__main__":
    tf.test.main()

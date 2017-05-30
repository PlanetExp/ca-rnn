"""test network slice"""

import tensorflow as tf
import numpy as np
from pprint import pprint
import os

import timeit


class DatasetTests(tf.test.TestCase):

    def testPrediction(self):
        # Class: (0, 1, ...)
        logits = tf.constant([[0.9, 0.5], [0.3, 0.6], [1, 88], [-134, 0], [1, 0], [1, 0], [1, 0], [1, 2]], tf.float32)
        labels = tf.constant([1, 1, 1, 1, 1, 1, 1, 1], tf.int32)
        pred = tf.nn.in_top_k(logits, labels, 1)

        # pred = tf.equal(1-tf.argmax(logits, 1), tf.cast(labels, tf.int64))
        a = tf.reduce_mean(tf.cast(pred, tf.float32))

        a2 = tf.reduce_sum(tf.cast(pred, tf.float32))
        ba = 8
        b = tf.div(a2, ba)

        s = [.5,.8,.9,.9,.6,.9]
        ss = np.sum(s) / len(s) * 100
        mm = np.mean(s) * 100
        assert ss == mm

        with tf.Session():

            print ("mm: %f" % (mm))
            print ("ss: %f" % (ss))

            print ("a.eval(): %f" % a.eval())
            print ("a2.eval(): %f" % b.eval())


if __name__ == "__main__":
    tf.test.main()

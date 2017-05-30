"""Test the Dataset class"""

import tensorflow as tf
import numpy as np

class DatasetTests(tf.test.TestCase):

    def testAccuracies(self):
        data = {'foo': [], 'bar': []}

        data['foo'].append(1)
        data['foo'].append(2)
        data['foo'].append(3)
        data['foo'].append(4)
        data['foo'].append(5)

        data['bar'].append(1)
        data['bar'].append(2)
        data['bar'].append(3)
        data['bar'].append(4)
        data['bar'].append(5)
        data['bar'].append(5)
        data['bar'].append(10)

        a = np.mean(data['foo'])
        b = np.mean(data['bar'])

        print (a,b)

        data = {'foo': [], 'bar': []}

        # tf.test(a,b)

        print (data)


if __name__ == "__main__":
    tf.test.main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

import input_pipeline


class InputPipeLineTest(tf.test.TestCase):

    def _record(self):
        '''
        9, 1, 1
        W  H  D
        '''
        # image_size = 9 * 1
        data = np.zeros((9, 1, 1))
        label = np.zeros(1)
        record = label.tobytes() + data.tobytes()

        # record = bytes(bytearray([label] + [channel] * image_size))
        expected = data
        return record, expected

    def testInput(self):

        labels = [0, 0, 0]
        records = [self._record(),
                   self._record(),
                   self._record()]
        contents = b"".join([record for record, _ in records])
        expected = [expected for _, expected in records]
        filename = os.path.join(self.get_temp_dir(), "test")
        open(filename, "wb").write(contents)

        with self.test_session() as sess:
            q = tf.FIFOQueue(99, [tf.string], shapes=())
            q.enqueue([filename]).run()
            q.close().run()
            result = input_pipeline.read_and_decode(q)

            for i in range(3):
                key, label, uint8inputs = sess.run([
                    result.key, result.label, result.uint8inputs])
                self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))
                self.assertEqual(labels[i], label)
                self.assertAllEqual(expected[i], uint8inputs)

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run([result.key, result.uint8inputs])


if __name__ == "__main__":
    tf.test.main()

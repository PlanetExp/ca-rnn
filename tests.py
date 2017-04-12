""""""

import tensorflow as tf
import numpy as np

import os
import utils


class InputTests(tf.test.TestCase):

	def testReader(self):
		filepath = os.path.join(self.get_temp_dir(), "test.bin")

		shape = (5, 5, 1)
		image = np.arange(25, dtype=np.int8).reshape(shape)
		label = np.ones((1), dtype=np.int8)
		batch_i = np.reshape(image, (1, 5, 5, 1))
		batch_l = np.reshape(label, (1, 1))
		utils._convert_to_tfrecords(batch_i, shape, batch_l, filepath)

		with self.test_session() as sess:
			q = tf.FIFOQueue(99, tf.string)
			q.enqueue([filepath]).run()
			q.close().run()
			tf.train.start_queue_runners(sess)
			i, l = utils.read_and_decode(q, shape)
			ii, ll = sess.run([i, l])

			self.assertAllEqual(ii, image)
			self.assertAllEqual(ll, label)

	def testGenerator(self):
		filepath = os.path.join(self.get_temp_dir(), "test.bin")
		shape = (20, 20, 1)
		num_examples = 4
		x, y = utils.generate_constrained_dataset(None, shape, num_examples, False, 0.5)
		utils._convert_to_tfrecords(x, shape, y, filepath)

		# print(x[3].reshape((5, 5)), y[3])

		with self.test_session() as sess:
			q = tf.FIFOQueue(99, tf.string)
			q.enqueue([filepath]).run()
			q.close().run()
			tf.train.start_queue_runners(sess)
			xx, yy = utils.read_and_decode(q, shape)
			yy = tf.squeeze(yy)

			for i in range(num_examples):
				image, label = sess.run([xx, yy])
				# print (image.reshape((5, 5)), label)
				self.assertAllEqual(x[i], image)
				self.assertAllEqual(y[i], label)

				# Add tests for correct label and image shape sizes

if __name__ == "__main__":
	tf.test.main()

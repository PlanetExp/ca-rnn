""""""

import tensorflow as tf
import numpy as np

import os
import utils
from conv_ca import ConvCA


class InputTests(tf.test.TestCase):

    # def testCkpts(self):
    #   """Testing saving and loading of checkoints"""
    #     path = os.path.join(self.get_temp_dir(), "train.ckpt")
    #     a = tf.constant([1])
    #     b = tf.constant([1])
    #     c = tf.Variable(a + b)
    #     init = tf.global_variables_initializer()
    #     saver = tf.train.Saver()
    #     with self.test_session() as sess:
    #         sess.run(init)
    #         for i in range(10):
    #             cc = sess.run(c)
    #             if i % 5 == 0:
    #                 s = saver.save(sess, path, i)
    #                 print ("saved in ", s, i)
    #         print ("var: ", cc)

    #         # ckpt_path = tf.train.latest_checkpoint(self.get_temp_dir())
    #         ckpt = tf.train.get_checkpoint_state(self.get_temp_dir())
    #         # tf.train.get_checkpoint_path()
    #         print (ckpt.model_checkpoint_path)
    #         # print (ckpt.get_checkpoint_path)
    #         if ckpt and ckpt.model_checkpoint_path:
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #             print("Model restored from ", ckpt.model_checkpoint_path)
    #             step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #         print (step)

    # def testTemplateNames(self):
    # """Testing template names"""
    #     def var_name(x, name):
    #         a = tf.get_variable(name,
    #                             shape=[2, 2],
    #                             initializer=tf.constant_initializer(1))
    #         return a


    #     var_name_foo = tf.make_template("foo", var_name, name='foo')
    #     a = var_name_foo(tf.constant([1]))
    #     var_name_bar = tf.make_template("bar", var_name, name='foo')
    #     b = var_name_bar(tf.constant([0]))

    #     print (a, b)
    #     with self.test_session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         aaa, bbb = sess.run([a, b])
    #         self.assertAllEqual(aaa, bbb)

    # def testScope(self):
    #     def s(name=None):
    #       with tf.variable_scope("scope") as vs:
    #           a = tf.get_variable(name, [1, 2, 3], initializer=tf.constant_initializer(1))
    #           vs.reuse_variables()
    #           b = tf.get_variable(name, [1, 2, 3], initializer=tf.constant_initializer(0))
    #       return a, b
    #     scope_copy = tf.make_template("scope", s)
    #     a, b = scope_copy("foo")
    #     aa, bb = scope_copy("foo")

    #     with self.test_session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         aaa, bbb = sess.run([a, bb])
    #         self.assertAllEqual(aaa, bbb)

    # def testTemplate(self):
    #     k = tf.placeholder(tf.float32, name="keep_prob")
    #     g = tf.Variable(0, trainable=False, name="global_step")
    #     m = ConvCA(g, k)
    #     it = tf.placeholder(tf.float32, [64, 9, 9, 1])
    #     iv = tf.placeholder(tf.float32, [64, 9, 9, 1])
    #     il = tf.placeholder(tf.int64, [64, ])
    #     t = tf.make_template("model", m)
    #     # model = t(g, k)
    #     valid = t(iv, il)
    #     train = t(it, il)

    #     data1 = np.ones((64, 9, 9, 1))
    #     data2 = np.zeros((64, 9, 9, 1))
    #     label = np.ones((64, ))

    #     with self.test_session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         a, b = sess.run([train.prediction, valid.prediction], feed_dict={k: 1.0, it: data1, iv: data2, il: label})
    #         self.assertEqual(train, valid)
    #         self.assertNotEqual(a, b)


    # def testReader(self):
    #   filepath = os.path.join(self.get_temp_dir(), "test.bin")

    #   shape = (5, 5, 1)
    #   image = np.arange(25, dtype=np.int8).reshape(shape)
    #   label = np.ones((1), dtype=np.int8)
    #   batch_i = np.reshape(image, (1, 5, 5, 1))
    #   batch_l = np.reshape(label, (1, 1))
    #   utils._convert_to_tfrecords(batch_i, shape, batch_l, filepath)

    #   with self.test_session() as sess:
    #       q = tf.FIFOQueue(99, tf.string)
    #       q.enqueue([filepath]).run()
    #       q.close().run()
    #       tf.train.start_queue_runners(sess)
    #       i, l = utils.read_and_decode(q, shape)
    #       ii, ll = sess.run([i, l])

    #       self.assertAllEqual(ii, image)
    #       self.assertAllEqual(ll, label)

    # def testGenerator(self):
    #   filepath = os.path.join(self.get_temp_dir(), "test.bin")
    #   shape = (20, 20, 1)
    #   num_examples = 4
    #   x, y = utils.generate_constrained_dataset(None, shape, num_examples, False, 0.5)
    #   utils._convert_to_tfrecords(x, shape, y, filepath)

    #   # print(x[3].reshape((5, 5)), y[3])

    #   with self.test_session() as sess:
    #       q = tf.FIFOQueue(99, tf.string)
    #       q.enqueue([filepath]).run()
    #       q.close().run()
    #       tf.train.start_queue_runners(sess)
    #       xx, yy = utils.read_and_decode(q, shape)
    #       yy = tf.squeeze(yy)

    #       for i in range(num_examples):
    #           image, label = sess.run([xx, yy])
    #           # print (image.reshape((5, 5)), label)
    #           self.assertAllEqual(x[i], image)
    #           self.assertAllEqual(y[i], label)

                # Add tests for correct label and image shape sizes

if __name__ == "__main__":
    tf.test.main()

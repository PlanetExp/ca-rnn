""""""

import tensorflow as tf
import numpy as np

import os
import utils
from conv_ca import ConvCA

from StringIO import StringIO
from matplotlib.pyplot import 


class InputTests(tf.test.TestCase):

    def testEmbeddings(self):
        """test projector and word embeddings"""

        slim = tf.contrib.slim

        data = np.random.randint(0, 2, size=(50, 1, 3, 3, 1))
        labels = np.random.randint(0, 2, size=(50, 1))

        x = tf.placeholder(tf.float32, [None, 3, 3, 1])
        y = tf.placeholder(tf.int32, [None])

        

        


        # net = slim.conv2d(x, 10, [3, 3])

        with tf.variable_scope("conv", initializer=tf.contrib.layers.xavier_initializer()):
            w = tf.Variable(tf.random_normal([3, 3, 1, 10], stddev=1.0), name="weights")
            b = tf.Variable(tf.zeros([10]), name="biases")
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            net = tf.nn.relu(conv + b)


        net = slim.conv2d(net, 1, [1, 1])
        y_ = slim.fully_connected(tf.reshape(net, [1, 9]), 2)



        loss = slim.losses.sparse_softmax_cross_entropy(y_, y)
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        pred = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_, y, 1), tf.float32))



        logdir = "tmp/test"
        path = os.path.join(logdir, "model.ckpt")
        projector = tf.contrib.tensorboard.plugins.projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = w.name
        # Link this tensor to its metadata file (e.g. labels).
        # embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
        writer = tf.summary.FileWriter(logdir)
        projector.visualize_embeddings(writer, config)

        saver = tf.train.Saver()


        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            acc = 0
            for i in range(20):
                feed_dict={x: data[i], y: labels[i]}
                sess.run(train_op, feed_dict=feed_dict)
                acc += sess.run(pred, feed_dict=feed_dict)
                step += 1

            print (acc / step)
            s = saver.save(sess, path, step)
            print ("saved in ", s, i)
            



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
    #     # m = ConvCA(g, k)
    #     it = tf.placeholder(tf.float32, [64, 9, 9, 1])
    #     iv = tf.placeholder(tf.float32, [64, 9, 9, 1])
    #     il = tf.placeholder(tf.int64, [64, ])
    #     t = tf.make_template("model", ConvCA)
    #     # model = t(g, k)
    #     valid = t(iv, il, g, k)
    #     train = t(it, il, g, k)

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

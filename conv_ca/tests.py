""""""

import tensorflow as tf
import numpy as np
from pprint import pprint
import os
import utils
import random
# from conv_ca import ConvCA

# from StringIO import StringIO





# def conv_ca_model(run_path, args=None):
#     """Links inputs and starts a training session and
#     performs logging at certain steps."""

#     tf.reset_default_graph()
#     sess = tf.Session()

#     # INPUTS
#     # ------

#     with tf.name_scope("inputs"):
#         # inputs, labels = input_pipeline(
#         #     DATADIR, FLAGS.batch_size,
#         #     shape=FLAGS.grid_shape,
#         #     num_threads=FLAGS.num_threads, istrain=True, name="train_pipe")

#         # inputs_valid, labels_valid = input_pipeline(
#         #     DATADIR, FLAGS.batch_size,
#         #     shape=FLAGS.grid_shape,
#         #     num_threads=FLAGS.num_threads, istrain=False, name="valid_pipe")

#         # Keep probability for dropout layer
#         keep_prob = tf.placeholder(tf.float32, name="keep_prob")
#         inputs_pl = tf.placeholder(
#             tf.float32, shape=[None, WIDTH, HEIGHT, DEPTH], name="inputs")
#         labels_pl = tf.placeholder(tf.int32, shape=[None], name="labels")

#     # GRAPH
#     # -----

#     # Make a template out of model to create two models that share the same graph
#     # and variables
#     shared_model = tf.make_template("model", ConvCA)
#     with tf.name_scope("train"):
#         train = shared_model(inputs_pl, labels_pl, keep_prob)
#     with tf.name_scope("valid"):
#         valid = shared_model(inputs_pl, labels_pl, keep_prob, istrain=False)

#     # Create writer to write summaries to file
#     writer = tf.summary.FileWriter(run_path, sess.graph)
#     writer.add_graph(sess.graph)

#     # EMBEDDING
#     # ---------

#     # This embedding attemps to reduce a batch of dimensions of the grid into a
#     # single point so that a single point represent a single grid that can be
#     # plotted into a 2/3 dimensinoal space with t-SNE.
#     if FLAGS.embedding:
#         embedding = tf.Variable(tf.zeros([FLAGS.num_embeddings,
#                                           valid.embedding_size]),
#                                 name="embedding")
#         assignment = embedding.assign(valid.embedding_input)
#         x_embedding, y_embedding = embedding_metadata(
#             DATADIR, FLAGS.grid_shape, FLAGS.num_embeddings)
#         setup_embedding_projector(embedding, writer)

#     # REST OF GRAPH
#     # -------------

#     # Create op to merge all summaries into one for writing to disk
#     merged_summary = tf.summary.merge_all()

#     # Save checkpoints for evaluations
#     saver = tf.train.Saver()
#     filename = os.path.join(run_path, "train.ckpt")

#     sess.run(tf.global_variables_initializer())

#     # RUN
#     # ---

#     # Create coordinator and start all threads from input_pipeline
#     # The queue will feed our model with data, so no placeholders are necessary
#     coord = tf.train.Coordinator()

#     step = 0
#     epoch = 0
#     start_time = time()
#     tot_running_time = start_time
#     try:
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#         # fetch embedding images for t-SNE visualization
#         # x_embedding_batch, y_embedding_batch = sess.run([x_embedding,
#         # y_embedding])

#         # Training loop
#         # -------------

#         # Training loop runs until coordinator have got a requested to stop
#         tot_accuracy = 0.0
#         tot_valid_accuracy = 0.0
#         accuracies = []
#         accuracies_valid = []
#         while not coord.should_stop():
#             # training
#             x_batch, y_batch = sess.run([inputs, labels])
#             feed_dict = {inputs_pl: x_batch,
#                          labels_pl: y_batch, keep_prob: FLAGS.dropout}
#             _, loss, accuracy = sess.run(
#                 [train.optimizer, train.loss, train.prediction], feed_dict)

#             # validation
#             x_batch, y_batch = sess.run([inputs_valid, labels_valid])
#             feed_dict = {inputs_pl: x_batch,
#                          labels_pl: y_batch, keep_prob: 1.0}
#             valid_accuracy, snap = sess.run(
#                 [valid.prediction, valid.activation_snapshot], feed_dict)

#             tot_accuracy += accuracy
#             tot_valid_accuracy += valid_accuracy
#             step += 1

#             # logging
#             # -------

#             if step % 10 == 0:
#                 summary = sess.run(merged_summary, feed_dict)
#                 writer.add_summary(summary, step)
#             elif step % 500 == 499:
#                 if FLAGS.run_metadata:
#                     # Optionally write run metadata into the checkoint file,
#                     # such as resource usage, memory consumption runtimes etc.
#                     run_options = tf.RunOptions(
#                         trace_level=tf.RunOptions.FULL_TRACE)
#                     run_metadata = tf.RunMetadata()
#                     summary = sess.run(
#                         merged_summary, feed_dict, run_options, run_metadata)
#                     writer.add_run_metadata(run_metadata, "step%d" % step)
#                     writer.add_summary(summary, step)

#                     # Create the Timeline object, and write it to a json
#                     # tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
#                     # ctf = tl.generate_chrome_trace_format()
#                     # with open("timeline.json", "w") as f:
#                     #     f.write(ctf)

#             # Request to close threads and stop at max_steps
#             if FLAGS.max_steps == step:
#                 coord.request_stop()

#             if step % FLAGS.log_frequency == 0:
#                 save_activation_snapshot(snap, step, args[0])
#                 current_time = time()
#                 duration = current_time - start_time
#                 start_time = current_time

#                 avg_accuracy = tot_accuracy / FLAGS.log_frequency
#                 avg_accuracy_valid = tot_valid_accuracy / FLAGS.log_frequency
#                 tot_accuracy = 0.0
#                 tot_valid_accuracy = 0.0

#                 # save logs of [[wall time, step, avg_accuracy]]
#                 accuracies.append([time(), step, avg_accuracy])
#                 accuracies_valid.append([time(), step, avg_accuracy])

#                 if avg_accuracy_valid > FLAGS.max_valid_accuracy:
#                     coord.request_stop()

#                 tag = "avg_accuracy/train"
#                 value = avg_accuracy
#                 s = tf.Summary(
#                     value=[tf.Summary.Value(tag=tag, simple_value=value)])
#                 writer.add_summary(s, step)

#                 tag = "avg_accuracy/valid"
#                 value = avg_accuracy_valid
#                 s = tf.Summary(
#                     value=[tf.Summary.Value(tag=tag, simple_value=value)])
#                 writer.add_summary(s, step)

#                 epoch += FLAGS.batch_size * FLAGS.log_frequency / FLAGS.num_examples
#                 examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
#                 sec_per_batch = float(duration / FLAGS.log_frequency)
#                 format_str = ("%s step %d/%d, epoch %.2f, loss: %.4f, avg. accuracy: %.4f (%.4f) "
#                               "(%.1fex/s; %.3fs/batch)")
#                 print(format_str % (datetime.now().strftime("%m/%d %H:%M:%S"),
#                                     step, FLAGS.max_steps, epoch, loss, avg_accuracy, avg_accuracy_valid,
#                                     examples_per_sec, sec_per_batch))

#                 progress = float(step / FLAGS.max_steps)
#                 estimated_duration = (
#                     FLAGS.max_steps * FLAGS.batch_size) * (1 - progress) / examples_per_sec
#                 t = timedelta(seconds=int(estimated_duration))
#                 format_str = "Estimated duration: %s (%.1f%%)"
#                 print(format_str % (str(t), progress * 100))

#             if step % 500 == 0:
#                 if FLAGS.embedding:
#                     # Assign embeddings to variable
#                     feed_dict = {inputs_pl: x_embedding,
#                                  labels_pl: y_embedding, keep_prob: 1.0}
#                     sess.run(assignment, feed_dict)

#                 # Save the model once on a while
#                 saver.save(sess, filename, global_step=step)

#     except Exception as e:
#         coord.request_stop(e)
#     finally:

#         # SAVE AND REPORT
#         # ---------------

#         save_path = saver.save(sess, filename, global_step=step)
#         print("Model saved in file: %s" % save_path)

#         coord.request_stop()
#         # Wait for threads to finish
#         coord.join(threads)

#         # Print some last stats
#         tot_duration = time() - tot_running_time
#         t = timedelta(seconds=int(tot_duration))
#         print("Total running time: %s" % t)
#         print("Layers: %d State dims: %d" %
#               (FLAGS.num_layers, FLAGS.state_size))

#         writer.close()
#         sess.close()
#         # report some stats so gridsearch can save them

#         if FLAGS.save_data:
#             dump = [{
#                 "timestamp": "%s" % datetime.now(),
#                 "runtime": "%s" % t,
#                 "training": accuracies,
#                 "validation": accuracies_valid,
#                 "batch_size": FLAGS.batch_size,
#                 "learning_rate": FLAGS.learning_rate,
#                 "layers": FLAGS.num_layers,
#                 "state_size": FLAGS.state_size,
#                 "num_examples": FLAGS.num_examples,
#                 "epochs": epoch,
#                 "dropout": FLAGS.dropout
#             }]
#             save_json(dump)







class InputTests(tf.test.TestCase):

    def testSlice(self):

        sess = tf.InteractiveSession()
        x = tf.constant(np.arange(18), tf.float32, shape=[2, 3, 3])
        print(x.eval())

        # s = tf.slice(i, [0, 0, 0, 0], [-1, 1, 3, 1])
        # s = tf.gather(i, [0, 0, 0, 0])
        idx = tf.constant([[[2]]])
        idx_flattened = tf.range(0, x.shape[0] * x.shape[1]) * x.shape[2] + idx
        y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                      idx_flattened)  # use flattened indices
        print("*****")
        print(y.eval())

        sess.close()


    # def testGenerator(self):
    #     x, y = utils.generate_constrained_dataset((20, 20), 8, stone_probability=0.4)

    #     # pprint(x)
    #     pprint(y)

    # def blobs(self):
    #     width = 5
    #     height = 5
    #     k = 1

    #     # def index(i, j)

    #     grid = np.zeros((width * height), dtype=np.uint8)

    #     for i in range(k):
    #         r = srandom.randint(len(grid))
    #         grid[r] = 1

    #     print(grid.shape)

        # filled = set()

    # ----------------------------
    # def testGenerator(self):
        # utils.save_emdedding_metadata("tmp/data/20x20", (20, 20, 1), 1024)

    # ----------------------------

    # def testPng(self):
    #     # import struct

    #     def write_png(buf, width, height):
    #         """ buf: must be bytes or a bytearray in Python3.x,
    #             a regular string in Python2.x.
    #         """
    #         import zlib
    #         import struct

    #         # reverse the vertical line order and add null bytes at the start
    #         width_byte_4 = width * 4
    #         raw_data = b''.join(b'\x00' + buf[span:span + width_byte_4]
    #                             for span in range((height - 1) * width_byte_4, -1, - width_byte_4))

    #         def png_pack(png_tag, data):
    #             chunk_head = png_tag + data
    #             return (struct.pack("!I", len(data)) +
    #                     chunk_head +
    #                     struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    #         return b''.join([
    #             b'\x89PNG\r\n\x1a\n',
    #             png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
    #             png_pack(b'IDAT', zlib.compress(raw_data, 9)),
    #             png_pack(b'IEND', b'')])


    #     def saveAsPNG(array, filename):
    #         import struct
    #         if any([len(row) != len(array[0]) for row in array]):
    #             raise ValueError ("Array should have elements of equal size")

    #         #First row becomes top row of image.
    #         flat = []
    #         map(flat.extend, reversed(array))
            
    #         #Big-endian, unsigned 32-byte integer.
    #         buf = b''.join([struct.pack('>I', ((0xffFFff & i32)<<8)|(i32>>24) )
    #                         for i32 in flat])   #Rotate from ARGB to RGBA.

    #         data = write_png(buf, len(array[0]), len(array))
    #         with open(filename, 'wb') as f:
    #             f.write(data)


    #     a = np.random.randint(0, 4, [29, 29, 3, 1], np.uint32)
    #     saveAsPNG(a, "asdf.png")


    # ----------------------------



    # def testEmbeddings(self):
    #     """test projector and word embeddings"""

    #     slim = tf.contrib.slim

    #     data = np.random.randint(0, 2, size=(50, 1, 3, 3, 1))
    #     labels = np.random.randint(0, 2, size=(50, 1))

    #     x = tf.placeholder(tf.float32, [None, 3, 3, 1])
    #     y = tf.placeholder(tf.int32, [None])

        
    # ----------------------------

        


    #     # net = slim.conv2d(x, 10, [3, 3])

    #     with tf.variable_scope("conv", initializer=tf.contrib.layers.xavier_initializer()):
    #         w = tf.Variable(tf.random_normal([3, 3, 1, 10], stddev=1.0), name="weights")
    #         b = tf.Variable(tf.zeros([10]), name="biases")
    #         conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    #         net = tf.nn.relu(conv + b)


    #     net = slim.conv2d(net, 1, [1, 1])
    #     y_ = slim.fully_connected(tf.reshape(net, [1, 9]), 2)



    #     loss = slim.losses.sparse_softmax_cross_entropy(y_, y)
    #     train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    #     pred = tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_, y, 1), tf.float32))



    #     logdir = "tmp/test"
    #     path = os.path.join(logdir, "model.ckpt")
    #     projector = tf.contrib.tensorboard.plugins.projector
    #     config = projector.ProjectorConfig()
    #     embedding = config.embeddings.add()
    #     embedding.tensor_name = w.name
    #     # Link this tensor to its metadata file (e.g. labels).
    #     # embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
    #     writer = tf.summary.FileWriter(logdir)
    #     projector.visualize_embeddings(writer, config)

    #     saver = tf.train.Saver()


    #     with self.test_session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         step = 0
    #         acc = 0
    #         for i in range(20):
    #             feed_dict={x: data[i], y: labels[i]}
    #             sess.run(train_op, feed_dict=feed_dict)
    #             acc += sess.run(pred, feed_dict=feed_dict)
    #             step += 1

    #         print (acc / step)
    #         s = saver.save(sess, path, step)
    #         print ("saved in ", s, i)
            
    # ----------------------------



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
    # ----------------------------

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
    # ----------------------------

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
    # ----------------------------

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
    # ----------------------------

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
    # ----------------------------

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

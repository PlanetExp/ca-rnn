"""Tensorflow model for exploring cellular automaton in convolutional neural
    networks.

This model uses Tensorboard for visualization though various summaries

Author: Frederick Heidrich
Supervisor: Dr. Frederic Maire
Date: 
"""
import tensorflow as tf
import numpy as np

import math

from datetime import datetime
from time import time, sleep
import os
import re

from utils import input_pipeline, maybe_generate_data


class FLAGS(object):
    """Temporary wrapper class for settings"""
    pass

# Parameters
FLAGS.state_size = 32
FLAGS.batch_size = 64
FLAGS.num_examples = 10000
FLAGS.num_classes = 2
# Set number of Cellular Automaton layers to stack.
FLAGS.num_layers = 6
# Set whether to reuse variables between CA layers or not.
FLAGS.reuse = True
FLAGS.learning_rate = 0.01
# Number of steps before printing logs.
FLAGS.log_frequency = 100
# Set maximum number of steps to train for
FLAGS.max_steps = 1000
# Start a number of threads per processor
FLAGS.num_threads = 4
FLAGS.data_dir = "tmp/data"
FLAGS.train_dir = "tmp/train"
FLAGS.valid_dir = "tmp/valid"

# Data parameters
width = 9
height = 9
depth = 1

# Construct hyperparameter string for our run based on settings
# (Example: "lr=1e-2,ca=3,state=64")
hparam_str = "lr=%.e,ca=%d,state=%d" % (FLAGS.learning_rate, FLAGS.num_layers, FLAGS.state_size)

# Generate dataset of (shape) if none exists in data_dir already
maybe_generate_data(FLAGS.data_dir, shape=(width, height, depth),
                    num_examples=FLAGS.num_examples)

# Flush train_dir for convenience
if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)


class ConvCA(object):
    """Defines convolution cellular automaton model"""
    def __init__(self, inputs, labels, global_step=None, keep_prob=None):
        self._inputs = inputs
        self._labels = labels
        self._global_step = global_step
        self._keep_prob = keep_prob

        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        self._create_prediction()

        # self._create_inference(inputs, keep_prob)
        # self._create_loss(self._logits, labels)
        # self._create_optimizer(self._loss, global_step)
        # self._create_prediction(self._logits, labels)

    def _create_inference(self):
        """Creates inference logits"""

        def _add_summaries(w, b, act):
            """Helper to add summaries"""
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
            tf.summary.scalar("sparsity", tf.nn.zero_fraction(act))

        def conv_layer(x, kernel,
                       initializer=None, name=None, scope=None):
            """Helper to create convolution layer and add summaries"""
            initializer = initializer or tf.truncated_normal_initializer(
                stddev=0.1, dtype=tf.float32)
            with tf.variable_scope(scope or name):
                w = tf.get_variable("weights", kernel,
                                    initializer=initializer, dtype=tf.float32)
                b = tf.get_variable("biases", [kernel[3]],
                                    initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(x, w,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME")
                act = tf.nn.relu(conv + b)
                _add_summaries(w, b, act)
            return act

        # Input convolution layer
        # -------------------------------------
        # Increase input depth from 1 to state_size
        conv1 = conv_layer(self._inputs, [3, 3, 1, FLAGS.state_size], name="input_conv")

        # Cellular Automaton module
        # -------------------------------------
        with tf.variable_scope("ca_conv") as scope:
            # List of all layer states
            state_layers = [conv1]
            for layer in range(FLAGS.num_layers):
                # Share weights between layers by marking scope with reuse
                if FLAGS.reuse and layer > 0:
                    scope.reuse_variables()

                conv_state = conv_layer(state_layers[-1],
                                        [3, 3, FLAGS.state_size, FLAGS.state_size],
                                        scope=scope)
                state_layers.append(conv_state)

        # Output module
        # -------------------------------------
        # reduce depth from state_size to 1
        initializer = tf.truncated_normal_initializer(
            stddev=1.0 / math.sqrt(float(FLAGS.state_size)), dtype=tf.float32)
        output = conv_layer(state_layers[-1], [1, 1, FLAGS.state_size, 1],
                            initializer=initializer, name="output_conv")

        # Add dropout layer
        dropout = tf.nn.dropout(output, self._keep_prob)

        # Softmax linear
        # -------------------------------------
        # Flatten output layer for classification
        with tf.variable_scope("softmax_linear"):
            # flatten to one dimension
            reshape = tf.reshape(dropout, [FLAGS.batch_size, -1])
            w = tf.get_variable("weights",
                [width * height, FLAGS.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.0),
                dtype=tf.float32)
            b = tf.get_variable("biases",
                [FLAGS.num_classes],
                initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.nn.xw_plus_b(reshape, w, b)
            _add_summaries(w, b, softmax_linear)

            self._logits = softmax_linear

    def _create_loss(self):
        """Create loss function"""
        with tf.name_scope("loss"):
            # Get rid of extra label dimension to [batch_size] and cast to int32
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self._labels, name="cross_entropy")
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

            tf.summary.scalar("loss", cross_entropy_mean)
            self._loss = cross_entropy_mean

    def _create_optimizer(self):
        """Create training optimizer"""
        with tf.name_scope("optimizer"):
            # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            minimize = optimizer.minimize(self._loss, global_step=self._global_step)
            self._optimizer = minimize

    def _create_prediction(self):
        with tf.name_scope("prediction"):    
            correct = tf.equal(tf.argmax(self._logits, 1), tf.cast(self._labels, tf.int64))
            # correct = tf.nn.in_top_k(self._logits, self._labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # print (correct)
            tf.summary.scalar("accuracy", accuracy)
            self._prediction = accuracy

    # Read-only properties
    @property
    def inference(self):
        return self._logits

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def prediction(self):
        return self._prediction


# test that num_classes are correct
#

# Testing steps
#

# Get a train data pipeline
inputs, labels = input_pipeline(FLAGS.data_dir, FLAGS.batch_size,
                                shape=(width, height, depth),
                                num_threads=FLAGS.num_threads, train=True)

# Attach an image summary to Tensorboard
# image_op = tf.summary.image('input', inputs, 3)

# Get validation data pipeline
# inputs_valid, labels_valid = input_pipeline(FLAGS.batch_size, train=False)

# Keep probability for dropout layer
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# Save a global step variable to keep track on steps
global_step = tf.Variable(0, trainable=False, name="global_step")

# Create model graph
# model = ConvCA(inputs, labels, global_step, keep_prob)
model = tf.make_template("train", ConvCA(inputs, labels, global_step, keep_prob))
test = model(inputs, labels, global_step, keep_prob)

# Initialize all variables that are trainable
init_op = tf.global_variables_initializer()

# Create up to merge all summaries into one for saving
merged_summary = tf.summary.merge_all()

# Start Tensorflow session
with tf.Session() as sess:
    sess.run(init_op)

    # Create writer to write summaries to file
    writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, hparam_str))
    writer.add_graph(sess.graph)

    # Create coordinator and start all threads from input_pipeline
    # The queue will feed our model with data, so no placeholders are necessary
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
   
    # Save checkpoints for evaluations
    # saver = tf.train.Saver(max_to_keep=1)
    
    # steps_per_epoch = num_examples / batch_size
    start_time = time()
    try:
        # Training loop until coordinator have requested to stop
        while not coord.should_stop():
            
            # Before run
            # --------------------------------
            # Train on data for
            # Take a snapshot of graph stats every 500th step and merge summary
            # and debug images. Merge summaries every 10th step.
            feed_dict = {keep_prob: 1.0}
            sess_run_args = [model.optimizer, model.loss, model.prediction, global_step]
            _, loss_value, accuracy, step = sess.run(sess_run_args, feed_dict)              
            
            if step % 10 == 0:
                summary = sess.run(merged_summary, feed_dict)
                writer.add_summary(summary, step)
            elif step % 500 == 499:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary = sess.run(merged_summary, feed_dict, run_options, run_metadata)
                writer.add_run_metadata(run_metadata, "step%d" % step)
                writer.add_summary(summary, step)

            # After run
            # --------------------------------
            # Request to close threads and stop at max_steps
            if FLAGS.max_steps == step:
                coord.request_stop()

            if step % FLAGS.log_frequency == 0:
                current_time = time()
                duration = current_time - start_time
                start_time = current_time

                # a = sess.run(model.inference, feed_dict)
                # print(a[:5])  # debug logits

                # Adjust log freq per time instead!
                #
                # sec_per_step = float(duration / step)

                # Save the model for evaluation
                # save_path = saver.save(sess, FLAGS.train_dir)
                # print("Model saved in file: %s" % save_path)

                # Restore model for evaluation
                # saver.restore(sess, FLAGS.train_dir)
                # print("Model restored.")

                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)

                format_str = ("%s: step %d, loss = %.4f, accuracy = %.3f (%.1f examples/sec; %.3f "
                    "sec/batch)")
                print (format_str % (datetime.now(), step, loss_value, accuracy,
                           examples_per_sec, sec_per_batch))


                # train and test accuracies
                #

                # plot
                #

    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        # Wait for threads to finish
        coord.join(threads)

    # Close file writer neatly
    writer.close()

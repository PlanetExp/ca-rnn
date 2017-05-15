"""Tensorflow model for exploring cellular automaton in convolutional neural
    networks.

The first convolutional layer maps the 3x3 Moore neighborhood input into a
vector of size "state_size".

input: [batch, with, height, depth]
input -> state_i
:
state_i = state_k
:
Optional number of layers k in 0-inf
state_k -> state_{k+1}
:
state_{k+1} -> logits

Basic Architecture:
conv input (3x3)
:
CA module of optional conv layers (3x3) that share weights
:
conv output (1x1)
:
softmax -> logits


This model uses Tensorboard for visualization through various summaries
Use Tensorboard with "tensorboard --logdir [logdir]"

Author: Frederick Heidrich
Supervisor: Dr. Frederic Maire
Date: 13/4/17
"""


from datetime import datetime, timedelta
from timeit import default_timer as timer
# from time import time
# from pprint import pprint

import sys
# import csv
# import json
import math
import os
# import re

import tensorflow as tf
import numpy as np
from utils import input_pipeline, maybe_generate_data, embedding_metadata
from random_walker import load_hdf5
from dataset import create_datasets


# Global container and accessor for flags and their values
FLAGS = tf.app.flags.FLAGS
# class FLAGS(object):
#     """Temporary wrapper class for settings"""
#     pass

# PARAMETERS
# ----------
# (flag_name, default_value, doc-string)
# determine hparam string
tf.app.flags.DEFINE_integer(
    "num_layers", 1, "Number of convolution layers to stack.")
tf.app.flags.DEFINE_integer(
    "state_size", 1,
    "Number of depth dimensions for each convolution layer in stack.")
tf.app.flags.DEFINE_integer(
    "run", 99, "Set subdirectory number to save logs to.")
tf.app.flags.DEFINE_integer(
    "batch_size", 256, "Set batch size per step")
tf.app.flags.DEFINE_float(
    "learning_rate", 0.1, "Set learning rate.")
tf.app.flags.DEFINE_integer(
    "log_frequency", 100, "Number of steps before printing logs.")
tf.app.flags.DEFINE_integer(
    "max_steps", 15000, "Set maximum number of steps to train for")
tf.app.flags.DEFINE_string(
    "data_dir", "data", "Directory of the dataset")
tf.app.flags.DEFINE_string(
    "train_dir", "tmp/train", "Directory to save train event files")
tf.app.flags.DEFINE_string(
    "checkpoint_dir", "tmp/train", "Directory to save checkpoints")
tf.app.flags.DEFINE_string(
    "logfile", "logfile", "Name of logfile")

NUM_CLASSES = 2
# Set whether to reuse variables between CA layers or not.
REUSE_VARIABLES = True
# If Leaky ReLU is used, set the rate of the leak
LRELU_RATE = 0.1
# Percent of dropout to apply to training process
DROPOUT = 1.0
# Fraction of dataset to split into test samples
TEST_SIZE = 0.2
# Set epsilon for Adam optimizer
EPSILON = 1.0
# Start a number of threads per processor.
NUM_THREADS = 1
# Size of grid: tuple of dim (width, height, depth)
GRID_SHAPE = (20, 20, 1)
# Whether or not to record embeddings for this run
FLAGS.embedding = False
# Whether to save image and label data per embedding
SAVE_EMBEDDING_METADATA = False
NUM_EMBEDDINGS = 32
# Whether to record run metadata (e.g. run times, memory consumption etc.)
# view these in either Tensorboard or save to json file with a
# tensorflow.python.client.timeline object and and view a web browser
SAVE_RUN_METADATA = False
# Saves snapshots of the layer activations in separate folder
SAVE_ACTIVATION_SNAPSHOT = False
SNAPSHOT_DIR = "tmp/snaps"


# Data parameters
WIDTH = GRID_SHAPE[0]
HEIGHT = GRID_SHAPE[1]
DEPTH = GRID_SHAPE[2]
PREFIX = str(WIDTH) + "x" + str(HEIGHT)
DATADIR = os.path.join(FLAGS.data_dir, PREFIX)
DATASET = os.path.join(FLAGS.data_dir, PREFIX, "connectivity.h5")


def _add_summaries(wights, bias, act):
    """Helper to add summaries"""
    tf.summary.histogram("weights", wights)
    tf.summary.histogram("biases", bias)
    tf.summary.histogram("activations", act)
    tf.summary.scalar("sparsity", tf.nn.zero_fraction(act))


def conv_layer(inputs, kernel,
               initializer=None, name=None, scope=None):
    """Helper to create convolution layer and add summaries"""
    initializer = initializer or tf.contrib.layers.xavier_initializer_conv2d()
    with tf.variable_scope(scope or name):
        wights = tf.get_variable(
            "weights", kernel, initializer=initializer, dtype=tf.float32)
        bias = tf.get_variable(
            "biases", [kernel[3]], initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(inputs, wights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        # act = tf.nn.relu(conv + bias)
        act = lrelu(conv + bias)  # add leaky ReLU
        # _add_summaries(w, b, act)
    return act


def lrelu(x, leak=LRELU_RATE, name="lrelu"):
    """Leaky ReLU implementation"""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def save_activation_snapshot(snap, step, path):
    """Save an image of each layer and each depths of the convolution
    layer. Each depth dimension with get its own width, height image saved.

    Args:
        snap: a np.ndarray of size [width, height, depth]
        step: the step the snapshot was taken at
        path: directory where to save the snaps
    """
    import matplotlib.image as image

    # print (snap)
    # print (len(snap))
    for i in enumerate(snap):
        # print (snap[i].shape)
        for j in range(FLAGS.state_size):
            img = snap[i][0, :, :, j].squeeze()
            # print (img.shape)
            string = "%d-s%d_l%d.png" % (step, j, i)
            image.imsave(os.path.join(path, string), img)


class ConvCA(object):
    def __init__(self, inputs, labels, keep_prob, istrain=True):
        """Creates inference logits and sets up a graph that can be reached
            through the properties inference, loss, optimizer and prediction
            respectively

        Args:
            inputs:
            labels:
            keep_prob: placeholder for dropout probability
            istrain: bool whether model is for training or testing, if train
                create an additional optimizer and loss function
        """
        # conv2d wants 3d data
        inputs = tf.reshape(inputs, [-1, WIDTH, HEIGHT, 1])
        # Input convolution layer
        # -----------------------
        # Increase input depth from 1 to state_size
        conv1 = conv_layer(
            inputs, [3, 3, 1, FLAGS.state_size], name="input_conv")

        # Cellular Automaton module
        # -------------------------
        self.activation_snapshot = []
        with tf.variable_scope("ca_conv") as scope:
            # List of all layer states
            state_layers = [conv1]
            for layer in range(FLAGS.num_layers):
                # Share weights between layers by marking scope with reuse
                if REUSE_VARIABLES and layer > 0:
                    scope.reuse_variables()

                conv_state = conv_layer(
                    state_layers[-1],
                    [3, 3, FLAGS.state_size, FLAGS.state_size],
                    scope=scope)
                state_layers.append(conv_state)
                # self.activation_snapshot.append(conv_state)

        # Output module
        # -------------
        # reduce depth from state_size to 1
        initializer = tf.truncated_normal_initializer(
            stddev=1.0 / math.sqrt(float(FLAGS.state_size)), dtype=tf.float32)
        output = conv_layer(state_layers[-1], [1, 1, FLAGS.state_size, 1],
                            initializer=initializer, name="output_conv")

        # Add dropout layer
        # -----------------
        dropout = tf.nn.dropout(output, keep_prob)

        # Force only local updates going forward
        # select only top row for regression
        # print (dropout)

        # self.debug = output  # debug local activations
        s = tf.slice(dropout, [0, 0, 0, 0], [-1, 20, 1, 1])
        # print (s)

        flattened = tf.reshape(s, [-1, WIDTH])
        # flattened = tf.reshape(dropout, [-1, WIDTH * HEIGHT])

        # Softmax linear
        # --------------
        with tf.variable_scope("softmax_linear"):
            weights = tf.get_variable(
                "weights", [WIDTH, NUM_CLASSES],
                # "weights", [WIDTH * HEIGHT, NUM_CLASSES],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
            bias = tf.get_variable(
                "biases", [NUM_CLASSES],
                initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(flattened, weights, bias)
            # _add_summaries(w, b, logits)
        self.inference = logits

        with tf.name_scope("prediction"):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.cast(labels, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.summary.scalar("accuracy", accuracy)
        self.prediction = accuracy

        if istrain:
            # Unique to train graph
            # Create loss function
            with tf.name_scope("loss"):
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels, name="cross_entropy"))
                # tf.summary.scalar("loss", cross_entropy)
            self.loss = cross_entropy

            # Create training optimizer
            with tf.name_scope("optimizer"):
                train_step = tf.train.AdamOptimizer(
                    FLAGS.learning_rate, beta1=0.9, beta2=0.999,
                    epsilon=EPSILON).minimize(
                        cross_entropy)
            self.optimizer = train_step
        else:
            # Unique to test graph
            if FLAGS.embedding:
                self.embedding_input = flattened
                self.embedding_size = WIDTH * HEIGHT


def conv_ca_model(run_path, args=None):
    """Links inputs and starts a training session and 
    performs logging at certain steps."""

    tf.reset_default_graph()
    sess = tf.Session()

    # Load datasets
    grids, connections, _ = load_hdf5(DATASET)
    grids = grids.reshape((-1, WIDTH, HEIGHT, 1))
    datasets = create_datasets(grids, connections, test_size=TEST_SIZE)
    # print (datasets.train.num_examples, datasets.test.num_examples)

    # Keep probability for dropout layer
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    inputs_pl = tf.placeholder(
        tf.float32, shape=[None, WIDTH, HEIGHT, DEPTH], name="inputs")
    labels_pl = tf.placeholder(tf.int32, shape=[None], name="labels")

    # Make a template out of model to create two models that
    # share the same graph and variables
    shared_model = tf.make_template("model", ConvCA)
    with tf.name_scope("train"):
        train = shared_model(inputs_pl, labels_pl, keep_prob)
    with tf.name_scope("test"):
        test = shared_model(inputs_pl, labels_pl, keep_prob, istrain=False)

    # model = ConvCA(inputs_pl, labels_pl, keep_prob)

    # Create writer to write summaries to file
    writer = tf.summary.FileWriter(run_path)
    writer.add_graph(sess.graph)

    # Create op to merge all summaries into one for writing to disk
    # merged_summary = tf.summary.merge_all()

    # Save checkpoints for evaluations
    saver = tf.train.Saver()
    filename = os.path.join(run_path, "train.ckpt")

    sess.run(tf.global_variables_initializer())

    step = 0
    start_time = timer()
    tot_running_time = start_time
    try:
        # tot_accuracy = 0.0
        # tot_valid_accuracy = 0.0
        accuracies = []
        test_accuracies = []
        losses = []
        while step <= FLAGS.max_steps:
            # training
            train_batch_x, train_batch_y = datasets.train.next_batch(
                FLAGS.batch_size)
            feed_dict = {inputs_pl: train_batch_x,
                         labels_pl: train_batch_y,
                         keep_prob: DROPOUT}
            _, loss, accuracy = sess.run(
                [train.optimizer, train.loss, train.prediction], feed_dict)
            accuracies.append(accuracy)
            losses.append(loss)

            # test
            test_batch_x, test_batch_y = datasets.test.next_batch(
                FLAGS.batch_size, shuffle_data=False)
            feed_dict = {inputs_pl: test_batch_x,
                         labels_pl: test_batch_y,
                         keep_prob: 1.0}
            test_accuracy = sess.run(
                [test.prediction], feed_dict)
            test_accuracies.append(test_accuracy)
            #  test_accuracy, snap = sess.run(
            #     [test.prediction, test.activation_snapshot], feed_dict)
            # test_accuracies.append(test_accuracy)

            # train
            # train_batch_x, train_batch_y = datasets.train.next_batch(
            #     FLAGS.batch_size)
            # feed_dict = {inputs_pl: train_batch_x,
            #              labels_pl: train_batch_y,
            #              keep_prob: DROPOUT}
            # _, loss, accuracy, temp = sess.run(
            #     [model.optimizer, model.loss, model.prediction, model.temp], feed_dict)
            # accuracies.append(accuracy)
            # losses.append(loss)

            # # test
            # test_batch_x, test_batch_y = datasets.test.next_batch(
            #     FLAGS.batch_size, shuffle_data=False)
            # feed_dict = {inputs_pl: test_batch_x,
            #              labels_pl: test_batch_y,
            #              keep_prob: 1.0}
            # test_accuracy = sess.run(
            #     [model.prediction], feed_dict)
            # test_accuracies.append(test_accuracy)

            step += 1

            # logging
            # -------
            # log(step, accuracies)

            # if step % 50 == 0:
            #     summary = sess.run(merged_summary, feed_dict)
            #     writer.add_summary(summary, step)

            if step % FLAGS.log_frequency == 0:

                # print (temp.shape)
                # print (temp[0].reshape(20,20)[:5, :5])
                # save_activation_snapshot(snap, step, args[0])
                current_time = timer()
                duration = current_time - start_time
                start_time = current_time

                # compute the last log_frequency number of averages
                avg_accuracy = 1 - np.mean(accuracies)
                avg_test_accuracy = 1 - np.mean(test_accuracies)
                avg_loss = np.mean(losses)

                write_scalar_summary(
                    writer, "avg_accuracy/train", avg_accuracy, step)
                write_scalar_summary(
                    writer, "avg_accuracy/test", avg_test_accuracy, step)
                write_scalar_summary(
                    writer, "avg_loss", avg_loss, step)

                accuracies = []
                test_accuracies = []
                losses = []

                epoch = datasets.train.epochs_completed
                examples_per_sec = (
                    FLAGS.log_frequency * FLAGS.batch_size / duration)
                sec_per_batch = float(duration / FLAGS.log_frequency)
                format_str = ("%s step %d/%d, epoch %.2f, loss: %.4f, "
                              "avg. mcr: %.4f (%.4f) "
                              "(%.1fex/s; %.3fs/batch)")
                print(format_str % (
                    datetime.now().strftime("%m/%d %H:%M:%S"),
                    step, FLAGS.max_steps, epoch, loss, avg_accuracy,
                    avg_test_accuracy, examples_per_sec, sec_per_batch))

                progress = float(step / FLAGS.max_steps)
                estimated_duration = (
                    (FLAGS.max_steps * FLAGS.batch_size) *
                    (1 - progress) / examples_per_sec)
                timed = timedelta(seconds=int(estimated_duration))
                format_str = "ETA: %s (%.1f%%)"
                print (format_str % (str(timed), progress * 100))

            if step % 1000 == 0:
                # Save the model periodically
                saver.save(sess, filename, global_step=step)
    finally:
        saver.save(sess, filename, global_step=step)

        tot_duration = timer() - tot_running_time
        timed = timedelta(seconds=int(tot_duration))
        print ("Total running time: %s" % timed)
        print ("Layers: %d, State dims: %d, Run: %d" %
            (FLAGS.num_layers, FLAGS.state_size, FLAGS.run))

        writer.close()
        sess.close()


def write_scalar_summary(writer, tag, value, step):
    """Helper to manually write a scalar summary to a writer"""
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def setup_embedding_projector(embedding, writer):
    """Sets up embedding projector in Tensorboard and links meta-data to
        a Tensorflow writer, that in turns saves the data to disk in a
        checkpoint file.

    Args:
        embedding: an embedding variable
        writer: a Tensorflow writer object
    """
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    if SAVE_EMBEDDING_METADATA:
        # Link this tensor to its metadata file (e.g. labels).
        # NOTE: Directory relative to where you start Tensorboard
        embedding_config.sprite.image_path = os.path.join(
            DATADIR, "sprite_1024.png")
        embedding_config.metadata_path = os.path.join(
            DATADIR, "labels_1024.tsv")
        # Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend([WIDTH, HEIGHT])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(
        writer, config)


class Logger(object):
    """Logger object that prints both to logfile and to stdout"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(FLAGS.logfile, ".log"), "a")

    def write(self, message):
        """writes a message both to terminal and to logfile"""
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        """Handles python3 flush"""
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def make_hparam_str(num_layers, state_size):
    """Construct hyperparameter string for our run based on settings
    Example: "lr=1e-2,ca=3,state=64
    """
    return "layers=%d,state=%d" % (num_layers, state_size)


def main(argv=None):  # pylint: disable=unused-argument
    """Runs main script by evaluating if data exists in data directory
    possibly generating new, generates a hparam string and runs training.

    Args:
        argv: 
            list of command line arguments that is run with the script
    """
    # Generate dataset of (shape) if none exists in data_dir already
    # maybe_generate_data(
    #     DATADIR,
    #     shape=GRID_SHAPE,
    #     stone_probability=0.45,
    #     num_examples=FLAGS.num_examples // FLAGS.num_files,
    #     num_files=FLAGS.num_files)

    # enable logging to file with print
    sys.stdout = Logger()

    hparam = make_hparam_str(
        FLAGS.num_layers, FLAGS.state_size)
    run_path = os.path.join(
        FLAGS.train_dir, PREFIX, hparam, "run" + str(FLAGS.run))

    print ("-" * 40 + "\n")
    print ("Starting run %d for %s" % (FLAGS.run, hparam))

    # Flush run_path for convenience
    if tf.gfile.Exists(run_path):
        tf.gfile.DeleteRecursively(run_path)
    tf.gfile.MakeDirs(run_path)

    args = []
    if SAVE_ACTIVATION_SNAPSHOT:
        snap_path = os.path.join(SNAPSHOT_DIR, hparam)
        # print(snap_path)
        if tf.gfile.Exists(snap_path):
            tf.gfile.DeleteRecursively(snap_path)
        tf.gfile.MakeDirs(snap_path)
        args.append(snap_path)

    # run training
    conv_ca_model(run_path, args)


if __name__ == "__main__":
    tf.app.run()

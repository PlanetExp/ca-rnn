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

import sys
import math
import os

import tensorflow as tf
import numpy as np
from utils import embedding_metadata
from dataset import load_hdf5, create_datasets


# Global container and accessor for flags and their values
FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags

# PARAMETERS
# ----------
# (flag_name, default_value, doc-string)

# Model parameters
flags.DEFINE_integer("num_layers", 1, "Number of convolution layers to stack.")
flags.DEFINE_integer("state_size", 1, "Number of depth dimensions for each convolution layer in stack.")
flags.DEFINE_float("learning_rate", 1e-4, "Set learning rate.")


# Dataset parameters
flags.DEFINE_integer("batch_size", 500, "Set batch size per step")
flags.DEFINE_integer("num_classes", 2, "...")
flags.DEFINE_integer("height", 8, "...")
flags.DEFINE_integer("width", 8, "...")
flags.DEFINE_integer("depth", 1, "...")
flags.DEFINE_float("test_fraction", 0.2, "...")


# Run parameters
flags.DEFINE_integer("log_frequency", 100, "Number of steps before printing logs.")
flags.DEFINE_integer("max_steps", 30000, "Set maximum number of steps to train for")
flags.DEFINE_boolean("debug", False, "Flush directories before every run and print more info.")


# Directories
flags.DEFINE_integer("run", 101, "Set subdirectory number to save logs to.")
flags.DEFINE_string("data_dir", "", "Directory of the dataset")
flags.DEFINE_string("result_dir", "", "Directory to save train event files")
flags.DEFINE_string("logfile", "logfile", "Name of logfile")

# Other options
flags.DEFINE_boolean("dense_logs", False, "Enables logging with carriage return.")

# Set whether to reuse variables between CA layers or not.
REUSE_VARIABLES = True
# If Leaky ReLU is used, set the rate of the leak
LRELU_RATE = 0.01
# Percent of dropout to apply to training process
DROPOUT = 0.9
# Fraction of dataset to split into test samples
TEST_SIZE = 0.5
# Set epsilon for Adam optimizer
EPSILON = 1e-8

# Saves snapshots of the layer activations in separate folder
SAVE_ACTIVATION_SNAPSHOT = False
SNAPSHOT_DIR = "tmp/snaps"

# Data parameters
PREFIX = str(FLAGS.height) + "x" + str(FLAGS.width)

def conv_layer(inputs,
               kernel,
               initializer=None,
               name=None, 
               scope=None):
    """Helper to create convolution layer and add summaries"""
    initializer = initializer or tf.contrib.layers.xavier_initializer_conv2d()
    with tf.variable_scope(scope or name):
        wights = tf.get_variable(
            "weights", kernel, initializer=initializer, dtype=tf.float32)  #  regularizer=tf.contrib.layers.l2_regularizer(0.5)
        bias = tf.get_variable(
            "biases", [kernel[3]], initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(inputs, wights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        # act = tf.nn.relu(conv + bias)
        act = lrelu(conv)  # add leaky ReLU
        # act = lrelu(conv + bias)  # add leaky ReLU
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
    def __init__(self, inputs, labels, keep_prob, global_step):
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
        # inputs = tf.reshape(inputs, [-1, HEIGHT, WIDTH, 1])
        # Input convolution layer
        # -----------------------
        # Increase input depth from 1 to state_size
        conv1 = conv_layer(
            inputs, [1, 1, 1, FLAGS.state_size], name="input_conv")

        # Cellular Automaton module
        # -------------------------
        # self.activation_snapshot = []
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
        # sliced = tf.slice(dropout, [0, 0, 0, 0], [-1, 1, FLAGS.width, 1])

        # Slice works just like numpy
        # print ("dropout.get_shape(): %s" % dropout.get_shape())

        # Shape: (batch, height, width, ...)
        sliced = dropout[:, :1, :, :]
        # print ("sliced.get_shape(): %s" % sliced.get_shape())

        # Shape: (batch, 1, width, ...) -- that is only the top row
        flattened = tf.reshape(sliced, [-1, FLAGS.width])
        # flattened = tf.reshape(dropout, [-1, WIDTH * HEIGHT])

        # Softmax linear
        # --------------
        with tf.variable_scope("softmax_linear"):
            weights = tf.get_variable(
                "weights", [FLAGS.width, FLAGS.num_classes],
                # "weights", [WIDTH * HEIGHT, NUM_CLASSES],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32)
            bias = tf.get_variable(
                "biases", [FLAGS.num_classes],
                initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(flattened, weights, bias)
            # _add_summaries(w, b, logits)
        self.inference = logits

        # Create loss function
        with tf.name_scope("loss"):
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels, name="cross_entropy"))

            # Add regularization
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # reg_constant = 0.01
            # loss = cross_entropy + reg_constant * np.sum(reg_losses)
        self.loss = cross_entropy

        # Create training optimizer
        with tf.name_scope("optimizer"):
            train_step = tf.train.AdamOptimizer(
                FLAGS.learning_rate, beta1=0.9, beta2=0.999,
                epsilon=EPSILON).minimize(
                    cross_entropy, global_step=global_step)
        self.optimizer = train_step

        with tf.name_scope("prediction"):
            # correct_prediction = tf.equal(
                # tf.argmax(logits, 1), tf.cast(labels, tf.int64))

            # in_top_k is the same as argmax 1 above
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.prediction = accuracy


def conv_ca_model(run_path, args=None):
    """Links inputs and starts a training session and 
    performs logging at certain steps."""

    tf.reset_default_graph()
    sess = tf.Session()

    # Load datasets
    examples, labels = load_hdf5(FLAGS.data_dir)
    examples = examples.reshape((-1, FLAGS.height, FLAGS.width, 1))
    datasets = create_datasets(examples, labels, test_fraction=FLAGS.test_fraction)

    # Keep probability for dropout layer
    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    inputs_pl = tf.placeholder(
        tf.float32, shape=[None, FLAGS.height, FLAGS.width, FLAGS.depth], name="inputs")
    labels_pl = tf.placeholder(tf.int32, shape=[None], name="labels")

    model = ConvCA(inputs_pl, labels_pl, keep_prob, global_step_tensor)

    # Create writer to write summaries to file
    writer = tf.summary.FileWriter(run_path)
    writer.add_graph(sess.graph)

    # Create op to merge all summaries into one for writing to disk
    # merged_summary = tf.summary.merge_all()

    # Save checkpoints for evaluations
    saver = tf.train.Saver()
    filename = os.path.join(run_path, "train.ckpt")

    # Continue training and evaluation session from previous run
    ckpt = tf.train.get_checkpoint_state(run_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("model restored from checkpoint")
        # take global_step from the checkpoint file path
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print ("Starting new run.")
        # _flush_directory(run_path)
        sess.run(tf.global_variables_initializer())

    start_time = timer()
    tot_running_time = start_time
    try:
        best_test_accuracy = 0.0
        data = {"train_accuracy": [], "test_accuracy": [], "losses": []}
        for step in range(1, FLAGS.max_steps):

            def _feed_dict(dropout=DROPOUT, test=False):
                if test:
                    input_batch, label_batch = datasets.test.next_batch(FLAGS.batch_size, shuffle_data=False)
                else:
                    input_batch, label_batch = datasets.train.next_batch(FLAGS.batch_size)
                return {inputs_pl: input_batch, labels_pl: label_batch, keep_prob: dropout}

            # train
            _, loss, train_accuracy, global_step = sess.run(
                [model.optimizer, model.loss, model.prediction, global_step_tensor], _feed_dict())

            # test
            test_accuracy = sess.run([model.prediction], _feed_dict(1.0, test=True))

            data["train_accuracy"].append(train_accuracy)            
            data["test_accuracy"].append(test_accuracy)
            data["losses"].append(loss)

            if step % FLAGS.log_frequency == 0:

                # print (temp.shape)
                # print (temp[0].reshape(20,20)[:5, :5])
                # save_activation_snapshot(snap, step, args[0])
                current_time = timer()
                duration = current_time - start_time
                start_time = current_time

                # compute averages
                avg_accuracy = np.mean(data["train_accuracy"]) * 100
                avg_test_accuracy = np.mean(data["test_accuracy"]) * 100
                avg_loss = np.mean(data["losses"]) * 100

                if avg_test_accuracy > best_test_accuracy:
                    best_test_accuracy = avg_test_accuracy

                write_scalar_summary(
                    writer, "avg_accuracy/train", avg_accuracy, global_step)
                write_scalar_summary(
                    writer, "avg_accuracy/test", avg_test_accuracy, global_step)
                write_scalar_summary(
                    writer, "avg_loss", avg_loss, global_step)

                # reset data
                data = {"train_accuracy": [], "test_accuracy": [], "losses": []}

                examples_per_sec = (
                    FLAGS.log_frequency * FLAGS.batch_size / duration)
                sec_per_batch = float(duration / FLAGS.log_frequency)
                progress = float(step / FLAGS.max_steps)
                estimated_duration = (
                    (FLAGS.max_steps * FLAGS.batch_size) *
                    (1 - progress) / examples_per_sec)
                timed = timedelta(seconds=int(estimated_duration))

                format_str = ("INFO:ETA: %s (step: %d): loss: %.4f, "
                              "global_step: %d, accuracy: %.3f, test: %.3f "
                              "(%.1fex/s; %.3fs/batch)")
                if FLAGS.dense_logs:
                    print(format_str % (str(timed), step, loss, global_step, avg_accuracy,
                        avg_test_accuracy, examples_per_sec, sec_per_batch), end="\r", flush=True)
                else:
                    print(format_str % (str(timed), step, loss, global_step, avg_accuracy,
                        avg_test_accuracy, examples_per_sec, sec_per_batch))


            if step % 1000 == 0:
                # Save the model periodically
                saver.save(sess, filename, global_step=step)
    finally:
        # Save run if stopped before max_steps
        if step != FLAGS.max_steps:
            saver.save(sess, filename, global_step=step)

        tot_duration = timer() - tot_running_time
        timed = timedelta(seconds=int(tot_duration))
        print ("\nTotal running time: %s" % timed)
        print ("Layers: %d, State dims: %d, Run: %d, lr: %.0e" %
            (FLAGS.num_layers, FLAGS.state_size, FLAGS.run, FLAGS.learning_rate))
        print ("Best accuracy (test): %f\n" % best_test_accuracy)

        writer.close()
        sess.close()


def write_scalar_summary(writer, tag, value, step):
    """Helper to manually write a scalar summary to a writer"""
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


class Logger(object):
    """Logger object that prints both to logfile and to stdout"""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, FLAGS.logfile + ".log"), "a")

    def write(self, message):
        """writes a message both to terminal and to logfile"""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Handles python3 flush"""
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def make_hparam_str(num_layers, state_size):
    """Construct hyperparameter string for our run based on settings
    Example: "lr=1e-2,ca=3,state=64
    """
    return "lr=%.0e,layers=%d,state=%d" % (FLAGS.learning_rate, num_layers, state_size)


def _flush_directory(directory):
    """Helper function to clean directories"""
    if tf.gfile.Exists(directory):
        tf.gfile.DeleteRecursively(directory)
    tf.gfile.MakeDirs(directory)


def main(argv=None):  # pylint: disable=unused-argument
    """Runs main script by evaluating if data exists in data directory
    possibly generating new, generates a hparam string and runs training.

    Args:
        argv:
            list of command line arguments that is run with the script
    """
    if FLAGS.result_dir == "":
        raise "No directory flag --result_dir to save results"

    hparam = make_hparam_str(
        FLAGS.num_layers, FLAGS.state_size)
    run_path = os.path.join(
        FLAGS.result_dir, PREFIX, hparam, "run" + str(FLAGS.run))

    if FLAGS.debug:
        _flush_directory(run_path)

    # enable logging to file with print
    if not tf.gfile.Exists(run_path):
        tf.gfile.MakeDirs(run_path)
    sys.stdout = Logger(run_path)

    print ("=" * 50 + "\n")
    print ("Starting run %d for %s" % (FLAGS.run, hparam))
    print ("Dataset: %s" % FLAGS.data_dir)
    print ("%s" % datetime.now())
    print ("-" * 50 + "\n")

    args = []
    if SAVE_ACTIVATION_SNAPSHOT:
        snap_path = os.path.join(SNAPSHOT_DIR, hparam)
        # print(snap_path)
        _flush_directory(snap_path)
        args.append(snap_path)

    # run training
    conv_ca_model(run_path, args)


if __name__ == "__main__":
    tf.app.run()

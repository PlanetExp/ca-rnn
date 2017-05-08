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
import tensorflow as tf
# import numpy as np

from datetime import datetime, timedelta
from time import time
from pprint import pprint

import json
import math
import os
# import re

from utils import input_pipeline, maybe_generate_data, embedding_metadata


class FLAGS(object):
    """Temporary wrapper class for settings"""
    pass


# PARAMETERS
# ----------
# Set number of Cellular Automaton layers to stack.
FLAGS.num_layers = 1
FLAGS.state_size = 1
FLAGS.batch_size = 128
# Number of examples to generate in the training set.
FLAGS.num_examples = 65537
# Number of files to split the training set into
FLAGS.num_files = 4
FLAGS.num_classes = 2
# Set whether to reuse variables between CA layers or not.
FLAGS.reuse = True
FLAGS.learning_rate = 0.1
# Percent of dropout to apply to training process
FLAGS.dropout = 1.0
# Number of steps before printing logs.
FLAGS.log_frequency = 100
# Set maximum number of steps to train for
FLAGS.max_steps = 1000
# Stop after an avg validation accuracy have been reached (e.g we know it
# can learn)
FLAGS.max_valid_accuracy = 0.98
# Set epsilon for Adam optimizer
FLAGS.epsilon = 1.0
# Start a number of threads per processor.
FLAGS.num_threads = 1
# Set various directories for files.
FLAGS.data_dir = "tmp/data"
FLAGS.train_dir = "tmp/train"
FLAGS.valid_dir = "tmp/valid"
FLAGS.checkpoint_dir = "tmp/train"
# Whether or not to restore checkpoint of training session.
# If true, then train for an additional amount of steps in max_steps.
FLAGS.restore = False
# Size of grid: tuple of dim (width, height, depth)
FLAGS.grid_shape = (20, 20, 1)
# Whether or not to record embeddings for this run
FLAGS.embedding = True
# Whether to save image and label data per embedding
FLAGS.embedding_metadata = False
FLAGS.num_embeddings = 32
# Whether to record run metadata (e.g. run times, memory consumption etc.)
# view these in either Tensorboard or save to json file with a
# tensorflow.python.client.timeline object and and view a web browser
FLAGS.run_metadata = False
FLAGS.snapshot = True
FLAGS.snap_dir = "tmp/snaps"
FLAGS.save_data = True

# Data parameters
WIDTH = FLAGS.grid_shape[0]
HEIGHT = FLAGS.grid_shape[1]
DEPTH = FLAGS.grid_shape[2]
PREFIX = str(WIDTH) + "x" + str(HEIGHT)
DATADIR = os.path.join(FLAGS.data_dir, PREFIX)


def _add_summaries(w, b, act):
    """Helper to add summaries"""
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    tf.summary.scalar("sparsity", tf.nn.zero_fraction(act))


def conv_layer(x, kernel,
               initializer=None, name=None, scope=None):
    """Helper to create convolution layer and add summaries"""
    initializer = initializer or tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope or name):
        w = tf.get_variable("weights", kernel,
                            initializer=initializer, dtype=tf.float32)
        b = tf.get_variable("biases", [kernel[3]],
                            initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(x, w,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        act = tf.nn.relu(conv + b)
        _add_summaries(w, b, act)
    return act


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
    for i in range(len(snap)):
        # print (snap[i].shape)
        for j in range(FLAGS.state_size):
            img = snap[i][0, :, :, j].squeeze()
            # print (img.shape)
            s = "%d-s%d_l%d.png" % (step, j, i)
            image.imsave(os.path.join(path, s), img)


class ConvCA(object):
    def __init__(self, inputs, labels, keep_prob, istrain=True):
        """Creates inference logits and sets up a graph that can be reached
            through the properties inference, loss, optimizer and prediction respectively

        Args:
            inputs:
            labels:
            keep_prob: placeholder for dropout probability
            istrain: bool whether model is for training or testing, if train
                create an additional optimizer and loss function
        """

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
                if FLAGS.reuse and layer > 0:
                    scope.reuse_variables()

                conv_state = conv_layer(
                    state_layers[-1], [3, 3, FLAGS.state_size, FLAGS.state_size], scope=scope)
                state_layers.append(conv_state)
                self.activation_snapshot.append(conv_state)

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

        flattened = tf.reshape(dropout, [-1, WIDTH * HEIGHT])

        # Softmax linear
        # --------------
        with tf.variable_scope("softmax_linear"):
            w = tf.get_variable("weights",
                                [WIDTH * HEIGHT, FLAGS.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable("biases",
                                [FLAGS.num_classes],
                                initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(flattened, w, b)
            _add_summaries(w, b, logits)
        self.inference = logits

        with tf.name_scope("prediction"):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.cast(labels, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
        self.prediction = accuracy

        if istrain:
            # Unique to train graph
            # Create loss function
            with tf.name_scope("loss"):
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels, name="cross_entropy"))
                tf.summary.scalar("loss", cross_entropy)
            self.loss = cross_entropy

            # Create training optimizer
            with tf.name_scope("optimizer"):
                train_step = tf.train.AdamOptimizer(
                    FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=FLAGS.epsilon).minimize(
                        cross_entropy)
            self.optimizer = train_step
        else:
            # Unique to test graph
            if FLAGS.embedding:
                self.embedding_input = flattened
                self.embedding_size = WIDTH * HEIGHT


def conv_ca_model(run_path, args=None):
    """Links inputs and starts a training session and performs logging at certain steps."""

    tf.reset_default_graph()
    sess = tf.Session()

    # INPUTS
    # ------

    with tf.name_scope("inputs"):
        inputs, labels = input_pipeline(
            DATADIR, FLAGS.batch_size,
            shape=FLAGS.grid_shape,
            num_threads=FLAGS.num_threads, istrain=True, name="train_pipe")

        inputs_valid, labels_valid = input_pipeline(
            DATADIR, FLAGS.batch_size,
            shape=FLAGS.grid_shape,
            num_threads=FLAGS.num_threads, istrain=False, name="valid_pipe")

        # Keep probability for dropout layer
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        inputs_pl = tf.placeholder(
            tf.float32, shape=[None, WIDTH, HEIGHT, DEPTH], name="inputs")
        labels_pl = tf.placeholder(tf.int32, shape=[None], name="labels")

    # GRAPH
    # -----

    # Make a template out of model to create two models that share the same graph
    # and variables
    shared_model = tf.make_template("model", ConvCA)
    with tf.name_scope("train"):
        train = shared_model(inputs_pl, labels_pl, keep_prob)
    with tf.name_scope("valid"):
        valid = shared_model(inputs_pl, labels_pl, keep_prob, istrain=False)

    # Create writer to write summaries to file
    writer = tf.summary.FileWriter(run_path, sess.graph)
    writer.add_graph(sess.graph)

    # EMBEDDING
    # ---------

    # This embedding attemps to reduce a batch of dimensions of the grid into a
    # single point so that a single point represent a single grid that can be
    # plotted into a 2/3 dimensinoal space with t-SNE.
    if FLAGS.embedding:
        embedding = tf.Variable(tf.zeros([FLAGS.num_embeddings,
                                          valid.embedding_size]),
                                name="embedding")
        assignment = embedding.assign(valid.embedding_input)
        x_embedding, y_embedding = embedding_metadata(
            DATADIR, FLAGS.grid_shape, FLAGS.num_embeddings)
        setup_embedding_projector(embedding, writer)

    # REST OF GRAPH
    # -------------

    # Create op to merge all summaries into one for writing to disk
    merged_summary = tf.summary.merge_all()

    # Save checkpoints for evaluations
    saver = tf.train.Saver()
    filename = os.path.join(run_path, "train.ckpt")

    sess.run(tf.global_variables_initializer())

    # RUN
    # ---

    # Create coordinator and start all threads from input_pipeline
    # The queue will feed our model with data, so no placeholders are necessary
    coord = tf.train.Coordinator()

    step = 0
    epoch = 0
    start_time = time()
    tot_running_time = start_time
    try:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # fetch embedding images for t-SNE visualization
        # x_embedding_batch, y_embedding_batch = sess.run([x_embedding,
        # y_embedding])

        # Training loop
        # -------------

        # Training loop runs until coordinator have got a requested to stop
        tot_accuracy = 0.0
        tot_valid_accuracy = 0.0
        accuracies = []
        accuracies_valid = []
        while not coord.should_stop():
            # training
            x_batch, y_batch = sess.run([inputs, labels])
            feed_dict = {inputs_pl: x_batch,
                         labels_pl: y_batch, keep_prob: FLAGS.dropout}
            _, loss, accuracy = sess.run(
                [train.optimizer, train.loss, train.prediction], feed_dict)

            # validation
            x_batch, y_batch = sess.run([inputs_valid, labels_valid])
            feed_dict = {inputs_pl: x_batch,
                         labels_pl: y_batch, keep_prob: 1.0}
            valid_accuracy, snap = sess.run(
                [valid.prediction, valid.activation_snapshot], feed_dict)

            tot_accuracy += accuracy
            tot_valid_accuracy += valid_accuracy
            step += 1

            # logging
            # -------

            if step % 10 == 0:
                summary = sess.run(merged_summary, feed_dict)
                writer.add_summary(summary, step)
            elif step % 500 == 499:
                if FLAGS.run_metadata:
                    # Optionally write run metadata into the checkoint file,
                    # such as resource usage, memory consumption runtimes etc.
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = sess.run(
                        merged_summary, feed_dict, run_options, run_metadata)
                    writer.add_run_metadata(run_metadata, "step%d" % step)
                    writer.add_summary(summary, step)

                    # Create the Timeline object, and write it to a json
                    # tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open("timeline.json", "w") as f:
                    #     f.write(ctf)

            # Request to close threads and stop at max_steps
            if FLAGS.max_steps == step:
                coord.request_stop()

            if step % FLAGS.log_frequency == 0:
                save_activation_snapshot(snap, step, args[0])
                current_time = time()
                duration = current_time - start_time
                start_time = current_time

                avg_accuracy = tot_accuracy / FLAGS.log_frequency
                avg_accuracy_valid = tot_valid_accuracy / FLAGS.log_frequency
                tot_accuracy = 0.0
                tot_valid_accuracy = 0.0

                # save logs of [[wall time, step, avg_accuracy]]
                accuracies.append([time(), step, avg_accuracy])
                accuracies_valid.append([time(), step, avg_accuracy])

                if avg_accuracy_valid > FLAGS.max_valid_accuracy:
                    coord.request_stop()

                tag = "avg_accuracy/train"
                value = avg_accuracy
                s = tf.Summary(
                    value=[tf.Summary.Value(tag=tag, simple_value=value)])
                writer.add_summary(s, step)

                tag = "avg_accuracy/valid"
                value = avg_accuracy_valid
                s = tf.Summary(
                    value=[tf.Summary.Value(tag=tag, simple_value=value)])
                writer.add_summary(s, step)

                epoch += FLAGS.batch_size * FLAGS.log_frequency / FLAGS.num_examples
                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)
                format_str = ("%s step %d/%d, epoch %.2f, loss: %.4f, avg. accuracy: %.4f (%.4f) "
                              "(%.1fex/s; %.3fs/batch)")
                print(format_str % (datetime.now().strftime("%m/%d %H:%M:%S"),
                                    step, FLAGS.max_steps, epoch, loss, avg_accuracy, avg_accuracy_valid,
                                    examples_per_sec, sec_per_batch))

                progress = float(step / FLAGS.max_steps)
                estimated_duration = (
                    FLAGS.max_steps * FLAGS.batch_size) * (1 - progress) / examples_per_sec
                t = timedelta(seconds=int(estimated_duration))
                format_str = "Estimated duration: %s (%.1f%%)"
                print(format_str % (str(t), progress * 100))

            if step % 500 == 0:
                if FLAGS.embedding:
                    # Assign embeddings to variable
                    feed_dict = {inputs_pl: x_embedding,
                                 labels_pl: y_embedding, keep_prob: 1.0}
                    sess.run(assignment, feed_dict)

                # Save the model once on a while
                saver.save(sess, filename, global_step=step)

    except Exception as e:
        coord.request_stop(e)
    finally:

        # SAVE AND REPORT
        # ---------------

        save_path = saver.save(sess, filename, global_step=step)
        print("Model saved in file: %s" % save_path)

        coord.request_stop()
        # Wait for threads to finish
        coord.join(threads)

        # Print some last stats
        tot_duration = time() - tot_running_time
        t = timedelta(seconds=int(tot_duration))
        print("Total running time: %s" % t)
        print("Layers: %d State dims: %d" %
              (FLAGS.num_layers, FLAGS.state_size))

        writer.close()
        sess.close()
        # report some stats so gridsearch can save them

        if FLAGS.save_data:
            dump = [{
                "timestamp": "%s" % datetime.now(),
                "runtime": "%s" % t,
                "training": accuracies,
                "validation": accuracies_valid,
                "batch_size": FLAGS.batch_size,
                "learning_rate": FLAGS.learning_rate,
                "layers": FLAGS.num_layers,
                "state_size": FLAGS.state_size,
                "num_examples": FLAGS.num_examples,
                "epochs": epoch,
                "dropout": FLAGS.dropout
            }]
            save_json(dump)


def save_json(data):
    """Save data dump to json file for visualization outside tensorboard"""
    filename = "data3.json"
    if os.path.isfile(filename) and os.stat(filename).st_size != 0:
        with open(filename) as f:
            d = json.load(f)
            d.extend(data)
            # pprint(d)
            data = d

    with open(filename, "w") as out:
        json.dump(data, out)
        print("json file saved: ", filename)


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
    if FLAGS.embedding_metadata:
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


def make_hparam_str(learning_rate, num_layers, state_size):
    """Construct hyperparameter string for our run based on settings
    Example: "lr=1e-2,ca=3,state=64
    """
    return "lr=%.0e,ca=%d,state=%d" % (learning_rate, num_layers, state_size)


def main(argv=None):
    """Runs main script by evaluating if data exists in data directory
    possibly generating new, generates a hparam string and runs training."""
    # Generate dataset of (shape) if none exists in data_dir already
    maybe_generate_data(
        DATADIR,
        shape=FLAGS.grid_shape,
        stone_probability=0.45,
        num_examples=FLAGS.num_examples // FLAGS.num_files,
        num_files=FLAGS.num_files)

    hparam = make_hparam_str(
        FLAGS.learning_rate, FLAGS.num_layers, FLAGS.state_size)
    run_path = os.path.join(FLAGS.train_dir, PREFIX, hparam)

    print("Starting run for %s" % hparam)

    # Flush run_path for convenience
    if tf.gfile.Exists(run_path):
        tf.gfile.DeleteRecursively(run_path)
    tf.gfile.MakeDirs(run_path)

    args = []
    if FLAGS.snapshot:
        snap_path = os.path.join(FLAGS.snap_dir, hparam)
        # print(snap_path)
        if tf.gfile.Exists(snap_path):
            tf.gfile.DeleteRecursively(snap_path)
        tf.gfile.MakeDirs(snap_path)
        args.append(snap_path)

    # run training
    conv_ca_model(run_path, args)


if __name__ == "__main__":
    tf.app.run()

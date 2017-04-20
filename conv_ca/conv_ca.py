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

from datetime import datetime
from time import time
import math
import os
# import re

from utils import input_pipeline, maybe_generate_data


class FLAGS(object):
    """Temporary wrapper class for settings"""
    pass

# Parameters
FLAGS.state_size = 32
FLAGS.batch_size = 64
# Number of examples to generate in the training set.
FLAGS.num_examples = 10000
FLAGS.num_classes = 2
# Set number of Cellular Automaton layers to stack.
FLAGS.num_layers = 4
# Set whether to reuse variables between CA layers or not.
FLAGS.reuse = True
FLAGS.learning_rate = 0.01
# Number of steps before printing logs.
FLAGS.log_frequency = 100
# Set maximum number of steps to train for
FLAGS.max_steps = 10000
# Start a number of threads per processor.
FLAGS.num_threads = 4
# Set various directories for files.
FLAGS.data_dir = "tmp/data"
FLAGS.train_dir = "tmp/train"
FLAGS.valid_dir = "tmp/valid"
FLAGS.checkpoint_dir = "tmp/train"
# Whether or not to restore checkpoint of training session.
# If true, then train for an additional amount of steps in max_steps.
FLAGS.restore = False

# Data parameters
width = 9
height = 9
depth = 1


class ConvCA(object):
    """Defines convolution cellular automaton model"""
    def __init__(self, inputs, labels, global_step, keep_prob, is_train=True):
        """Creates inference logits and sets up a graph

        Args:
            inputs: inputs of shape [batch, width, height, depth]
            labels: labels of shape [batch]
            is_train: bool, if true also construct loss and optimizer
        """
        self._global_step = global_step
        self._keep_prob = keep_prob

        self._logits = self._create_inference(inputs)
        self._prediction = self._create_prediction(self._logits, labels)

        if is_train:
            self._loss = self._create_loss(self._logits, labels)
            self._optimizer = self._create_optimizer(self._loss)

    def _create_inference(self, inputs):
        """Construct computational graph.

        Args:
            inputs: inputs of shape [batch, width, height, depth]

        Returns:
            logits
        """
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
                                    initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(x, w,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME")
                act = tf.nn.relu(conv + b)
                _add_summaries(w, b, act)
            return act

        # Input convolution layer
        # -----------------------
        # Increase input depth from 1 to state_size
        conv1 = conv_layer(inputs, [3, 3, 1, FLAGS.state_size], name="input_conv")

        # Cellular Automaton module
        # -------------------------
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

        # Output module
        # -------------
        # reduce depth from state_size to 1
        initializer = tf.truncated_normal_initializer(
            stddev=1.0 / math.sqrt(float(FLAGS.state_size)), dtype=tf.float32)
        output = conv_layer(state_layers[-1], [1, 1, FLAGS.state_size, 1],
                            initializer=initializer, name="output_conv")

        # Add dropout layer
        # -----------------
        dropout = tf.nn.dropout(output, self._keep_prob)

        flattened = tf.reshape(dropout, [FLAGS.batch_size, -1])

        # Softmax linear
        # --------------
        with tf.variable_scope("softmax_linear"):
            w = tf.get_variable("weights",
                [width * height, FLAGS.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.0),
                dtype=tf.float32)
            b = tf.get_variable("biases",
                [FLAGS.num_classes],
                initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.nn.xw_plus_b(flattened, w, b)


            self.embedding_input = flattened
            self.embedding_size = height * width



            _add_summaries(w, b, softmax_linear)
        return softmax_linear

    def _create_loss(self, logits, labels):
        # Create loss function
        with tf.name_scope("loss"):
            # Get rid of extra label dimension to [batch_size] and cast to int32
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name="cross_entropy")
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

            tf.summary.scalar("loss", cross_entropy_mean)
        return cross_entropy_mean

    def _create_optimizer(self, loss):
        # Create training optimizer
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0)
            # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            minimize = optimizer.minimize(loss, global_step=self._global_step)
        return minimize

    def _create_prediction(self, logits, labels):
        with tf.name_scope("prediction"):
            correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
            # correct = tf.nn.in_top_k(self._logits, self._labels, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            # print (correct)
            tf.summary.scalar("accuracy", accuracy)
        return accuracy

    # Read-only properties
    # --------------------
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

    # End of model
    # ============


def run_training(run_path):
    """Run training and validation steps

    Args:
        run_path: file path where run data is stored. It is based on
            hyperparameters.
    """
    # Attach an image summary to Tensorboard
    # tf.summary.image('input', inputs_valid, 3)

    # Keep probability for dropout layer
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # Save a global step variable to keep track on steps
    global_step = tf.Variable(0, trainable=False, name="global_step")

    # Create model graph
    # --------------------------------------------------------------
    # Here we create two separate graphs that share the same weights
    # one for validation and one for train, and feed them separate
    # input streams. tf.make_template creates a copy of graph with
    # same variables.
    # model = ConvCA(global_step, keep_prob)
    shared_model = tf.make_template("model", ConvCA)
    with tf.name_scope("train"):
        inputs, labels = input_pipeline(
            FLAGS.data_dir, FLAGS.batch_size,
            shape=(width, height, depth),
            num_threads=FLAGS.num_threads, train=True)
        train = shared_model(inputs, labels, global_step, keep_prob)
    with tf.name_scope("valid"):
        inputs_valid, labels_valid = input_pipeline(
            FLAGS.data_dir, FLAGS.batch_size,
            shape=(width, height, depth),
            num_threads=FLAGS.num_threads, train=False)
        valid = shared_model(inputs_valid, labels_valid, global_step, keep_prob)

    # Initialize all variables that are trainable
    init_op = tf.global_variables_initializer()
    # Create op to merge all summaries into one for writing to disk
    merged_summary = tf.summary.merge_all()





    # Start Tensorflow session
    sess = tf.Session()


    embedding = tf.Variable(tf.zeros([FLAGS.batch_size, train.embedding_size]), name="embedding")
    assignment = embedding.assign(train.embedding_input)


    # Save checkpoints for evaluations
    saver = tf.train.Saver(max_to_keep=1)
    filename = os.path.join(run_path, "train.ckpt")

    sess.run(init_op)

    # Create writer to write summaries to file
    writer = tf.summary.FileWriter(run_path)
    writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    print(embedding.name)
    # Link this tensor to its metadata file (e.g. labels).
    # embedding.metadata_path = os.path.join(FLAGS.checkpoint_dir, 'metadata.tsv')
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)




    # Create coordinator and start all threads from input_pipeline
    # The queue will feed our model with data, so no placeholders are necessary
    coord = tf.train.Coordinator()
    


    step = 0
    start_time = time()
    try:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Training loop
        # ==============================================================
        # Training loop runs until coordinator have got a requested to stop
        tot_accuracy = 0.0
        tot_valid_accuracy = 0.0
        while not coord.should_stop():
            
            # Run training
            # --------------------------------
            feed_dict = {keep_prob: 1.0}
            sess_run_args = [train.optimizer, train.loss, train.prediction, valid.prediction]
            _, loss_value, accuracy, valid_accuracy = sess.run(sess_run_args, feed_dict)

            tot_accuracy += accuracy
            tot_valid_accuracy += valid_accuracy
            step += 1

            # After run and logging
            # --------------------------------
            # Take a snapshot of graph stats every 500th step and merge summary
            # and debug images. Merge summaries every 10th step.
            if step % 10 == 0:
                summary = sess.run(merged_summary, feed_dict)
                writer.add_summary(summary, step)
            elif step % 500 == 499:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary = sess.run(merged_summary, feed_dict, run_options, run_metadata)
                writer.add_run_metadata(run_metadata, "step%d" % step)
                writer.add_summary(summary, step)

            # Request to close threads and stop at max_steps
            if FLAGS.max_steps == step:
                coord.request_stop()

            if step % FLAGS.log_frequency == 0:
                current_time = time()
                duration = current_time - start_time
                start_time = current_time

                avg_accuracy        = tot_accuracy          / FLAGS.log_frequency
                avg_valid_accuracy  = tot_valid_accuracy    / FLAGS.log_frequency
                tot_accuracy        = 0.0
                tot_valid_accuracy  = 0.0




                # print validation accuracy for log (this is also saved in Tensorboard)
                valid_accuracy = sess.run(valid.prediction, feed_dict={keep_prob: 1.0})
                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)

                format_str = ("%s step %d, loss: %.4f, avg. accuracy: %.5f (%.5f) "
                              "(%.1f examples/sec; %.3f sec/batch)")
                print (format_str % (datetime.now().strftime("%y-%m-%d %H:%M:%S"),
                                     step, loss_value, avg_accuracy, avg_valid_accuracy,
                                     examples_per_sec, sec_per_batch))

            if step % 500 == 0:
                

                sess.run(assignment, feed_dict=feed_dict)




                # Save the model once on a while
                saver.save(sess, filename, global_step=step)










    except Exception as e:
        coord.request_stop(e)
    finally:
        save_path = saver.save(sess, filename, global_step=step)
        print("Model saved in file: %s" % save_path)


        coord.request_stop()
        # Wait for threads to finish
        coord.join(threads)
    # Close file writer neatly
    writer.close()

    sess.close()







def eval_once(valid, feed_dict):




    saver = tf.train.Saver()



    with tf.Session() as sess:

        # Restore model parameters form checkpoint
        checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Model restored from ", checkpoint.model_checkpoint_path)
        #     # Take steps from the file path
            global_step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        #     FLAGS.max_steps = FLAGS.max_steps + step
        else:
            print ("no checkpoint file found.")
            return


        # number of steps to evaluate for
        num_steps = 1000


        coord = tf.train.Coordinator()
        try:
            # Extend current queue runner threads to feed evaluation data into the validation graph
            threads = []
            for queue_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue_runner.create_threads(sess, coord=coord, daemon=True, start=True))


            num_iterations = int(math.ceil(num_steps / FLAGS.batch_size))
            step = 0
            tot_accuracy = 0
            while step < num_iterations and not coord.should_stop():
                accuracy = sess.run(valid.prediction, feed_dict=feed_dict)
                tot_accuracy += accuracy
                step += 1

            avg_accuracy = tot_accuracy / num_iterations
            print ("validation accuracy: ", avg_accuracy)



        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def make_hparam_str(learning_rate, num_layers, state_size):
    """Construct hyperparameter string for our run based on settings
    Example: "lr=1e-2,ca=3,state=64
    """
    return "lr=%.0e,ca=%d,state=%d" % (learning_rate, num_layers, state_size)

def main(argv=None):
    # Generate dataset of (shape) if none exists in data_dir already
    maybe_generate_data(FLAGS.data_dir, shape=(width, height, depth),
                        num_examples=FLAGS.num_examples)

    hparam = make_hparam_str(FLAGS.learning_rate, FLAGS.num_layers, FLAGS.state_size)
    run_path = os.path.join(FLAGS.train_dir, hparam)

    print ("Starting run for %s" % hparam)

    # Flush run_path for convenience
    if tf.gfile.Exists(run_path):
        tf.gfile.DeleteRecursively(run_path)
    tf.gfile.MakeDirs(run_path)

    run_training(run_path)


if __name__ == "__main__":
    # Runs main with args defined in tf.app.FLAGS
    tf.app.run()

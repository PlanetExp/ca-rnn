import os
import sys
import json
import random

import click
import tensorflow as tf


from conv_model import ConvModel
from conv_ca import conv_ca_model


DIR = os.path.dirname(os.path.realpath(__file__))

FLAGS = tf.app.flags.FLAGS

# Hyper parameter settings


# Model configurations
# FLAGS.DEFINE_string("name", "model", "Unique name of model.")
# FLAGS.DEFINE_integer("num_layers", 1, "Number of convolution layers to stack.")
# FLAGS.DEFINE_integer("state_size", 1, "Number of depth dimensions for each convolution layer in stack.")

# FLAGS.DEFINE_integer("batch_size", 256, "Set batch size per step")
# FLAGS.DEFINE_float("learning_rate", 0.001, "Set learning rate.")
# FLAGS.DEFINE_boolean('best', False, 'Force to use the best known configuration')

# FLAGS.DEFINE_boolean('reuse_variables', True, 'Reuse variables in stacked layers')

# FLAGS.DEFINE_float("lrelu_rate", 0.02, "Set leaky ReLU leak rate.")
# FLAGS.DEFINE_float("epsilon", 1e-8, "Epsilon for Adam optimizer.")

# FLAGS.DEFINE_integer("width", 8, "...")
# FLAGS.DEFINE_integer("height", 8, "...")
# FLAGS.DEFINE_integer("depth", 1, "...")
# FLAGS.DEFINE_integer("num_classes", 2, "...")


# # Environment configuration
# FLAGS.DEFINE_integer("run", 99, "Set subdirectory number to save logs to.")
# FLAGS.DEFINE_integer("log_frequency", 100, "Number of steps before printing logs.")
# FLAGS.DEFINE_integer("max_steps", 15000, "Set maximum number of steps to train for")
# FLAGS.DEFINE_string("data_dir", "data", "Directory of the dataset")
# FLAGS.DEFINE_string("train_dir", "../results", "Directory to save train event files")
# FLAGS.DEFINE_string("checkpoint_dir", "../results", "Directory to save checkpoints")
# FLAGS.DEFINE_string("logfile", "logfile", "Name of logfile")
# FLAGS.DEFINE_boolean("load_checkoint", False, "Whether or not to load checkpoint and continue training from last step.")
# FLAGS.DEFINE_string("result_dir", DIR, "Name of logfile")

# FLAGS.DEFINE_boolean('debug', True, 'Debug mode')
# FLAGS.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')


tf.app.flags.DEFINE_integer(
    "num_layers", 1, "Number of convolution layers to stack.")
tf.app.flags.DEFINE_integer(
    "state_size", 1,
    "Number of depth dimensions for each convolution layer in stack.")
tf.app.flags.DEFINE_integer(
    "run", 101, "Set subdirectory number to save logs to.")
tf.app.flags.DEFINE_integer(
    "batch_size", 500, "Set batch size per step")
tf.app.flags.DEFINE_float(
    "learning_rate", 0.001, "Set learning rate.")
tf.app.flags.DEFINE_integer(
    "log_frequency", 250, "Number of steps before printing logs.")
tf.app.flags.DEFINE_integer(
    "max_steps", 30000, "Set maximum number of steps to train for")
tf.app.flags.DEFINE_string(
    "data_dir", "data", "Directory of the dataset")
tf.app.flags.DEFINE_string(
    "train_dir", "../results", "Directory to save train event files")
tf.app.flags.DEFINE_string(
    "checkpoint_dir", "../results", "Directory to save checkpoints")
tf.app.flags.DEFINE_string(
    "logfile", "logfile", "Name of logfile")
tf.app.flags.DEFINE_boolean(
    "load_checkoint", True,
    "Whether or not to load checkpoint and continue training from last step.")


class Logger(object):
    """Logger object that prints both to logfile and to stdout"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(FLAGS.logfile + ".log", "a")

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
    # return "lr=%.0e,layers=%d,state=%d" % (FLAGS.FALGS.learning_rate, num_layers, state_size)
    pass 


# @click.command()
# @click.option('gridsearch')
def main(argv=None):
    # config = FLAGS.__flags.copy()  # hack

    # model = ConvModel(config)
    # model.train()

    # FLAGS = FLAGS.FLAGS

    sys.stdout = Logger()

    hparam = make_hparam_str(FLAGS.num_layers, FLAGS.state_size)
    run_path = os.path.join(FLAGS.train_dir, hparam, "run" + str(FLAGS.run))

    print ("=" * 50 + "\n")
    # print ("Starting run %d for %s" % (FLAGS.run, hparam))
    print ("Dataset: %s" % "test")
    print ("%s" % datetime.now().strftime("%m/%d %H:%M:%S"))
    print ("-" * 50 + "\n")

    run_path = "tmp/ast"
    conv_ca_model(run_path)


if __name__ == '__main__':
    tf.app.run()

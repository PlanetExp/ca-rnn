import os
import sys
import json
import random

import click
import tensorflow as tf


from conv_model import ConvModel


DIR = os.path.dirname(os.path.realpath(__file__))
print (DIR)
FLAGS = tf.app.flags

# Hyper parameter settings


# Model configurations
FLAGS.DEFINE_string("name", "model", "Unique name of model.")
FLAGS.DEFINE_integer("num_layers", 1, "Number of convolution layers to stack.")
FLAGS.DEFINE_integer("state_size", 1, "Number of depth dimensions for each convolution layer in stack.")

FLAGS.DEFINE_integer("batch_size", 256, "Set batch size per step")
FLAGS.DEFINE_float("learning_rate", 0.04, "Set learning rate.")
FLAGS.DEFINE_boolean('best', False, 'Force to use the best known configuration')

FLAGS.DEFINE_boolean('reuse_variables', True, 'Reuse variables in stacked layers')

FLAGS.DEFINE_float("lrelu_rate", 0.02, "Set leaky ReLU leak rate.")
FLAGS.DEFINE_float("epsilon", 0.9, "Epsilon for Adam optimizer.")

FLAGS.DEFINE_integer("width", 8, "...")
FLAGS.DEFINE_integer("height", 8, "...")
FLAGS.DEFINE_integer("depth", 1, "...")
FLAGS.DEFINE_integer("num_classes", 2, "...")


# Environment configuration
FLAGS.DEFINE_integer("run", 99, "Set subdirectory number to save logs to.")
FLAGS.DEFINE_integer("log_frequency", 100, "Number of steps before printing logs.")
FLAGS.DEFINE_integer("max_steps", 15000, "Set maximum number of steps to train for")
FLAGS.DEFINE_string("data_dir", "../../data/connectivity_8x8.h5", "Directory of the dataset")
FLAGS.DEFINE_string("train_dir", "tmp/train", "Directory to save train event files")
FLAGS.DEFINE_string("checkpoint_dir", "tmp/train", "Directory to save checkpoints")
FLAGS.DEFINE_string("logfile", "logfile", "Name of logfile")
FLAGS.DEFINE_boolean("load_checkoint", False, "Whether or not to load checkpoint and continue training from last step.")
FLAGS.DEFINE_string("result_dir", DIR, "Name of logfile")

FLAGS.DEFINE_boolean('debug', True, 'Debug mode')
FLAGS.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')


# @click.command()
# @click.option('gridsearch')
def main(_):
    config = FLAGS.FLAGS.__flags.copy()  # hack

    # model = ConvModel(config)
    # model.train()
    os.system("python conv_ca.py")


if __name__ == '__main__':
    tf.app.run()

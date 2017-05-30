"""stub"""
from base_model import BaseModel
import tensorflow as tf
import numpy as np
import math

from dataset import load_hdf5, create_datasets


class ConvModel(BaseModel):

    def set_properties(self):
        self.width = self.config['width']
        self.height = self.config['height']
        self.depth = self.config['depth']
        self.state_size = self.config['state_size']
        self.num_classes = self.config['num_classes']
        self.num_layers = self.config['num_layers']
        self.learning_rate = self.config['learning_rate']
        self.reuse_variables = self.config['reuse_variables']

        print (self.data_dir)

        grids, connections, _ = load_hdf5(self.data_dir)
        grids = grids.reshape((-1, self.width, self.height, 1))
        self.datasets = create_datasets(grids, connections, test_size=0.2)

    def build_graph(self, graph):
        with graph.as_default() as g:
            # Builds the graph as far as is required for running the network forward to make predictions.

            self.init_op = tf.global_variables_initializer()
            self.global_step = tf.Variable(0, trainable=False, name="global_step")

            self.inputs = tf.placeholder(
                tf.float32, shape=[None, self.width, self.height, self.depth], name="inputs")
            self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")

            # inputs = tf.reshape(inputs, [-1, self.width, self.height, 1])

            # Input convolution layer
            # Increase input depth from 1 to state_size
            conv1 = conv_layer(
                self.inputs, [3, 3, 1, self.state_size], name="input_conv")

            # Cellular Automata-like convolution stack
            with tf.variable_scope("layers") as scope:
                # List of all layer states
                state_layers = [conv1]
                for layer in range(self.num_layers):
                    # Share weights between layers by marking scope with reuse
                    if self.reuse_variables and layer > 0:
                        scope.reuse_variables()

                    conv_state = conv_layer(
                        state_layers[-1],
                        [3, 3, self.state_size, self.state_size],
                        initializer=tf.truncated_normal_initializer(stddev=1.0, dtype=tf.float32),
                        scope=scope)
                    state_layers.append(conv_state)

            # Output module
            # reduce depth from state_size to 1
            initializer = tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(float(self.state_size)), dtype=tf.float32)
            output = conv_layer(state_layers[-1], [1, 1, self.state_size, 1],
                initializer=initializer, name="output_conv")

            # Force only local updates going forward
            # select only top row for regression
            sliced = tf.slice(output, [0, 0, 0, 0], [-1, 1, self.height, 1])

            flattened = tf.reshape(sliced, [-1, self.width])
            # flattened = tf.reshape(dropout, [-1, self.width * self.height])

            # Softmax linear
            with tf.variable_scope("softmax_linear"):
                weights = tf.get_variable(
                    "weights", [self.width, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
                bias = tf.get_variable(
                    "biases", [self.num_classes],
                    initializer=tf.zeros_initializer())
                logits = tf.nn.xw_plus_b(flattened, weights, bias)
                self._logits = logits

                assert self._logits.graph is graph

            # Adds to the inference graph the ops required to generate loss.
            with tf.name_scope("loss"):
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self._logits, labels=self.labels, name="cross_entropy"))
                self._loss = cross_entropy

            # Adds to the loss graph the ops required to compute and apply gradients.
            with tf.name_scope("training"):
                train_step = tf.train.AdamOptimizer(
                    self.learning_rate, beta1=0.9, beta2=0.999,
                    epsilon=self.epsilon).minimize(self._loss, global_step=self.global_step)
                self._training = train_step

            with tf.name_scope("prediction"):
                correct_prediction = tf.equal(
                    tf.argmax(self._logits, 1), tf.cast(self.labels, tf.int64))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self._prediction = accuracy
            return g

    def _feed_dict(self, test=False):
        if test:
            input_batch, label_batch = self.datasets.test.next_batch(self.batch_size, shuffle_data=False)
        else:
            input_batch, label_batch = self.datasets.train.next_batch(self.batch_size)
        return {self.inputs: input_batch, self.labels: label_batch}

    def optimize(self):
        # with self.sess:

        global_step = tf.train.global_step(self.sess, self.global_step)
        steps_per_epoch = self.datasets.train.num_examples // self.batch_size

        # train, test, loss
        data = {"train_accuracy": [], "test_accuracy": [], "losses": []}
        for step in range(steps_per_epoch):
            _, loss, accuracy = self.sess.run([self.training, self.loss, self.prediction], self._feed_dict())
            test_accuracy = self.sess.run([self.prediction], self._feed_dict(test=True))
            
            data["train_accuracy"].append(accuracy)            
            data["test_accuracy"].append(test_accuracy)
            data["losses"].append(loss)

        avg_accuracy = np.mean(data[0])
        avg_test_accuracy = np.mean(data)
        avg_loss = np.mean(data)

        write_scalar_summary(self.writer, "avg_accuracy/train", avg_accuracy, global_step)
        write_scalar_summary(self.writer, "avg_accuracy/test", avg_test_accuracy, global_step)
        write_scalar_summary(self.writer, "avg_loss", avg_loss, global_step)

        print (avg_accuracy, avg_test_accuracy)

        data = {"train_accuracy": [], "test_accuracy": [], "losses": []}
        # test_accuracies = []
        # losses = []

    def get_best_config(self):
        return {} 

    @property
    def inference(self):
        """"""
        return self._logits

    @property
    def loss(self):
        """"""
        return self._loss

    @property
    def training(self):
        return self._training

    @property
    def prediction(self):
        return self._prediction


def write_scalar_summary(writer, tag, value, step):
    """Helper to manually write a scalar summary to a writer"""
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def conv_layer(
        inputs, kernel, initializer=None, name=None, scope=None):
    """Helper to create convolution layer and add summaries"""
    initializer = initializer or tf.contrib.layers.xavier_initializer_conv2d()
    with tf.variable_scope(scope or name):
        wights = tf.get_variable(
            "weights", kernel, initializer=initializer, dtype=tf.float32)
        bias = tf.get_variable(
            "biases", [kernel[3]], initializer=tf.constant_initializer(0.01))
        conv = tf.nn.conv2d(inputs, wights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        # act = tf.nn.relu(conv + bias)
        act = lrelu(conv + bias)  # add leaky ReLU
        return act


def lrelu(x, leak=0.01, name="lrelu"):
    """Leaky ReLU implementation"""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)




        

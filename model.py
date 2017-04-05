import tensorflow as tf

import input_pipeline
import utils
import os
import sys
import re

FLAGS = tf.app.flags.FLAGS
'''Global flags interface for command line arguments

Definition:
    flag_name, default_value, doc_string

Usage:
    python <file> [--flag_name=value]
    python <file> --help
        to list them all

Uses argparse module internally.
See: https://www.tensorflow.org/api_docs/python/tf/flags
'''
tf.app.flags.DEFINE_integer('batch_size', 128,
                            '''Size of batch.''')
tf.app.flags.DEFINE_integer('state_size', 16,
                            '''Size of cell states.''')
tf.app.flags.DEFINE_integer('rnn_size', 5,
                            '''Size of rnn.''')
tf.app.flags.DEFINE_integer('learning_rate', 0.001,
                            '''Learning rate.''')
tf.app.flags.DEFINE_integer('num_classes', 2,
                            '''Number of label classes.''')
tf.app.flags.DEFINE_integer('num_layers', 3,
                            '''Number of generations or layers of cellular automaton to generate.''')
tf.app.flags.DEFINE_string('data_dir', 'tmp/data',
                           '''Path to the data directory.''')
tf.app.flags.DEFINE_string('cell_name', 'lstm',
                           '''Name of cell to use.''')
tf.app.flags.DEFINE_boolean('reuse', True,
                            '''Whether to reuse weight variables or not.''')

# Global constants
INPUTS_SHAPE = input_pipeline.INPUTS_SHAPE
NUM_EXAMPLES = input_pipeline.NUM_EXAMPLES

'''
version 0.1
    + wrote doc string comments for most functionality
    + implemented ca-model base class to inherit other classes
    + Renamed inputs and data to more general names
    + generation is now layers (layers of CA)
    + added ca in 1d for LSTM RNN
    + moved inputs to separate file to use with both train and eval
    - removed model class in favor to scrips
    - removed @properties in favor of external creation (less complexity)
'''


# def _variable_on_cpu(name, shape, initializer):
#     """Helper to create a Variable stored on CPU memory.
#     We do this for distributed training in train.py

#     Args:
#         name: name of the variable
#         shape: list of ints
#         initializer: initializer for Variable
#     Returns:
#         Variable Tensor
#     """
#     with tf.device('/cpu:0'):
#         var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
#     return var


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on Tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def maybe_generate_data():
    '''Generate testing and training data if none exists in data_dir'''
    dest_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Log hook to measure progress
    # TODO: not in use
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Generating %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    # generate training batches
    # constrained
    num_files = 2
    stone_probability = 0.45
    filenames = ['data_batch_%d.tfrecords' % i for i in range(1, num_files + 1)]

    for filename in filenames:
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print('%s not found - generating...' % filename)
            utils.generate_constrained_dataset(filepath, _progress, **{
                'num_examples': NUM_EXAMPLES,
                'stone_probability': stone_probability,
                'shape': INPUTS_SHAPE})
            print()
            statinfo = os.stat(filepath)
            print('Successfully generated', filename, statinfo.st_size, 'bytes.')

    # generate testing batches
    # random
    # TODO
    filename = 'test_batch.tfrecords'
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        print('%s not found - generating...' % filename)
        # utils.generate_dataset(filepath, _progress, **{
        utils.generate_constrained_dataset(filepath, _progress, **{
            'num_examples': NUM_EXAMPLES,
            'stone_probability': stone_probability,
            'shape': INPUTS_SHAPE})
        print()
        statinfo = os.stat(filepath)
        print('Successfully generated', filename, statinfo.st_size, 'bytes.')


class BaseModelCA(object):
    '''
    CA model base class containing inputs streams and the shared static methods:
        - loss: loss function
        - optimizer: training op
        - prediction: prediction op

    Inference is implemented in each model inheriting this base class.
    It should contain:
        - inference: inference op, returning logits

    Models contains two separate input streams, one distorted input for training
    and one for testing.

    Notes:
        @staticmethod decorator marks a function as a regular function not inheriting its
        base class. See: http://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
    '''
    def inference(self, inputs):
        assert 'Inference not implemented.'

    def loss(self, logits, labels):
        # loss function
        with tf.name_scope('loss'):
            # Compute moving averages of all individual losses and total loss
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy')

            # cross_entropy_mean = tf.reduce_mean(cross_entropy)
            # Add all losses to graph
            # tf.add_to_collection('losses', cross_entropy_mean)
            # The total loss is defined as the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            # loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            loss = tf.reduce_mean(cross_entropy)
        return loss
    
    def optimizer(self, total_loss, global_step):
        # optimizer train_op

        # TRY: lr = tf.train.exponential_decay

        # Generate moving averages of all losses
        # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        # losses = tf.get_collection('losses')
        # loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        # for l in losses + [total_loss]:
        #     # Name each loss as '(raw)' and name the moving average version of the loss
        #     # as the original loss name.
        #     tf.summary.scalar(l.op.name + ' (raw)', l)
        #     tf.summary.scalar(l.op.name, loss_averages.average(l))

        # Compute gradients
        # Creates an average of all losses before applying minimize to optimizer op
        # with tf.control_dependencies([loss_averages_op]):
        #     optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        #     grads = optimizer.compute_gradients(total_loss)  # minimize() 1/2

        # # Apply gradients
        # # minimize() 2/2
        # apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Execute operations defined as arguments to control_dependencies before what's
        # inside its name scope.
        # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        #     train_op = tf.no_op(name='train')

        # simple version of above
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss, global_step=global_step)  
        return train_op

    def prediction(self, logits, labels):
        '''Evaluate the quality of the logits at predicting the label.
        
        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
                range [0, NUM_CLASSES).

        Returns:
            A scalar int64 tensor with the number of examples (out of batch_size)
            that were predicted correctly.
        '''
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        with tf.name_scope('prediction'):
            # tf.nn.in_top_k does the same thing as equal(argmax)
            # correct = tf.nn.in_top_k(logits, labels, 1)
            labels = tf.cast(labels, tf.int64)
            correct = tf.equal(tf.argmax(logits, 1), labels)
            # prediction = tf.reduce_mean(tf.cast(correct, tf.float32))
        # Return the number of true entries.
        return correct

    def train_inputs(self):
        '''Distorted input stream for training.'''
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
        inputs, labels = input_pipeline.train_inputs(
            data_dir=data_dir,
            batch_size=FLAGS.batch_size)
        return inputs, labels

    def inputs(self, eval_data):
        '''Clean input stream for evaluation and testing.'''
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
        inputs, labels = input_pipeline.inputs(
            eval_data=eval_data,
            data_dir=data_dir,
            batch_size=FLAGS.batch_size)
        return inputs, labels


class RecurrentCA(BaseModelCA):
    '''Class implementing Cellular Automaton in Recurrent Neural Network and
        testing various cell functions.

        It supports cell selection to try different variations of cells.width

        Currently only supports LSTMCell

        WIP
    '''
    def inference(self, inputs):
        '''Created inference for graph.

        Args:
            inputs: of shape [batch x width x height x depth]

        Returns:
            Logits
        '''
        with tf.name_scope('inference'):

            # Cell selection to try different versions of cells
            # currently only supports LSTMCell
            additional_cell_args = {}
            # Standard LSTMCell implementation
            # which has a split tuple of internal states, one cell state and one hidden state
            # State comes in the format of LSTMStateTuple NamedTuple which requires some 
            # special work to unfold.
            if FLAGS.cell_name == 'lstm':
                cell_fn = tf.contrib.rnn.LSTMCell
            # Vanilla RNN cell 
            elif FLAGS.cell_name == 'rnn':
                cell_fn = tf.contrib.rnn.RNNCell
                # additional_cell_args.update({'tied': True})
            else:
                # Raise Exception of cell is not supported
                raise Exception('Unsupported cell_name: {}'.format(FLAGS.cell_name))

            # Create cell from selection with optional additional cell arguments keywords
            cell = cell_fn(FLAGS.state_size, **additional_cell_args)

            # Remove all dimensions that are 1 for simplicity
            inputs = tf.squeeze(inputs)

            # Create zero_states and save in LSTMStateTuple object
            c_zero = tf.zeros([FLAGS.batch_size, FLAGS.state_size])
            h_zero = tf.zeros([FLAGS.batch_size, FLAGS.state_size])
            zero_state = tf.contrib.rnn.LSTMStateTuple(c_zero, h_zero)
            # Create zero states for all unfolded inputs
            states = [zero_state] * FLAGS.rnn_size

            
            # Split inputs into list of single inputs to unfold over
            inputs_as_list = tf.split(inputs, int(inputs.get_shape()[1]), axis=1)
            outputs = []

            # Layer unfolding module
            with tf.variable_scope('lstm_grid') as scope:
                for l in range(FLAGS.num_layers):
                    # Unfold cells over inputs
                    for i, input_ in enumerate(inputs_as_list):
                        if FLAGS.reuse and i > 0:
                            scope.reuse_variables()

                        # Create a strided slice to extract neighbours around cell
                        index = i + 2
                        neigh_states = states[max(0, index - 3): index]

                        # concat neighboring states
                        # c_state = tf.concat([neigh_states[j].c for j in range(len(neigh_states))], 1)
                        # h_state = tf.concat([neigh_states[j].h for j in range(len(neigh_states))], 1)
                        # BUG: state_size grows linerarly per generation

                        # element-wise multiply neighbor states, essentially a conv
                        # Store result in new LSTMStateTuple object
                        (c_states, h_states) = [list(a) for a in zip(*neigh_states)]
                        c_state = tf.reduce_sum(c_states, 0)
                        h_state = tf.reduce_sum(h_states, 0)
                        state = tf.contrib.rnn.LSTMStateTuple(c_state, h_state)

                        # Get output and new cell state from cell
                        (output, new_state) = cell(input_, state, scope)

                        # Store new state in unfolded state grid
                        states[i] = new_state

                        # If this layer is the last layer, store all outputs in a list
                        if l == FLAGS.num_layers - 1:  # last generation
                            outputs.append(output)

            # Output module
            # Softmax to two classes
            with tf.variable_scope('rnn'):
                # V matrices
                softmax_w = tf.get_variable('softmax_w', [FLAGS.rnn_size * FLAGS.state_size, FLAGS.num_classes])
                softmax_b = tf.get_variable('softmax_b', [FLAGS.num_classes])

            with tf.name_scope('softmax'):
                output = tf.concat(outputs, 1)
                logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        return logits
    

class ConvolutionalCA(BaseModelCA):
    def inference(self, inputs):
        '''Builds inference for the graph.

        Args:
            inputs: inputs returned from train_inputs() or inputs() of size
                [batch x width x height x depth]

        Returns:
            logits
        '''
        # Group scoped ops together in graph
        with tf.name_scope('inference'):

            # TRY: for optimization: weight decay

            dtype = tf.float32
            initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=dtype)
            # initializer = tf.contrib.layers.xavier_initializer()
            activation = tf.nn.relu
            # activation = tf.nn.tanh

            # Input convolution layer
            with tf.variable_scope('conv1') as scope:
                # Increase input depth from 1 to state_size
                kernel = tf.get_variable('weights', [3, 3, 1, FLAGS.state_size],
                                         initializer=initializer, dtype=dtype)
                biases = tf.get_variable('biases', [FLAGS.state_size],
                                         initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv1 = activation(pre_activation, name=scope.name)
                # add activation summary to visualize activations in Tensorboard
                _activation_summary(conv1)

            # Cellular Automaton module
            with tf.variable_scope('ca_conv') as scope:
                # List of all layer states
                state_layers = [conv1]
                for layer in range(FLAGS.num_layers):
                    # Mark variables to be reused within the same name scope
                    # after layer 0
                    if FLAGS.reuse and layer > 0:
                        scope.reuse_variables()

                    # layer input is state_size to state_size
                    kernel = tf.get_variable('weights', [3, 3, FLAGS.state_size, FLAGS.state_size],
                                             initializer=initializer, dtype=dtype)
                    biases = tf.get_variable('biases', [FLAGS.state_size],
                                             initializer=tf.constant_initializer(0.1))
                    # Previous layer as input
                    conv = tf.nn.conv2d(state_layers[-1], kernel, strides=[1, 1, 1, 1], padding='SAME')
                    pre_activation = tf.nn.bias_add(conv, biases)
                    conv_state = activation(pre_activation, name=scope.name + '_%s' % str(layer))
                    state_layers.append(conv_state)
                    _activation_summary(conv_state)

            # Output module
            with tf.variable_scope('output_conv'):
                # reduce depth from state_size to 1
                kernel = tf.get_variable('weights', [3, 3, FLAGS.state_size, 1],
                                         initializer=tf.truncated_normal_initializer(
                                             stddev=1.0 / tf.square(float(FLAGS.state_size)),
                                             dtype=dtype),
                                         dtype=dtype)
                biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(conv_state, kernel, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                output = activation(pre_activation, name=scope.name)
                _activation_summary(output)

            # Flatten output layer for classification
            # Note:
            #   tf.sparse_softmax_cross_entropy_with_logits loss function applies softmax internally
            #   So no need to do that here.
            with tf.variable_scope('softmax_linear'):
                # flatten to one dimension
                reshape = tf.reshape(output, [FLAGS.batch_size, -1])
                # input_width = INPUTS_SHAPE[0]
                # input_height, INPUTS_SHAPE[1]
                softmax_w = tf.get_variable('weights', [INPUTS_SHAPE[0] * INPUTS_SHAPE[1], FLAGS.num_classes],
                                            initializer=tf.truncated_normal_initializer(stddev=0.0), dtype=dtype)
                softmax_b = tf.get_variable('biases', [FLAGS.num_classes], initializer=tf.constant_initializer(0.0))
                softmax_linear = tf.nn.xw_plus_b(reshape, softmax_w, softmax_b)
                logits = softmax_linear
                _activation_summary(softmax_linear)

            return logits

import tensorflow as tf

import input_pipeline
import utils
import os
import sys

# Global flags interface for command line arguments
# Uses the argparse module internally
# See: https://www.tensorflow.org/api_docs/python/tf/flags
FLAGS = tf.app.flags.FLAGS

# flag_name, default_value, doc_string
# Usage: python <file> [--flag_name=value]
tf.app.flags.DEFINE_integer('batch_size', 128,
                            '''Size of batch.''')
tf.app.flags.DEFINE_integer('state_size', 16,
                            '''Size of cell states.''')
tf.app.flags.DEFINE_integer('rnn_size', 5,
                            '''Size of rnn.''')
tf.app.flags.DEFINE_integer('learning_rate', 0.1,
                            '''Learning rate.''')
tf.app.flags.DEFINE_integer('num_classes', 2,
                            '''Number of label classes.''')
tf.app.flags.DEFINE_integer('num_layers', 10,
                            '''Number of generations or layers of cellular automata to generate.''')
tf.app.flags.DEFINE_string('data_dir', 'tmp/data',
                           '''Path to the data directory.''')
tf.app.flags.DEFINE_string('cell_name', 'lstm',
                           '''Name of cell to use.''')
tf.app.flags.DEFINE_boolean('reuse', True,
                           '''Whether to reuse weight variables or not.''')

# Global constants
INPUTS_SIZE = input_pipeline.INPUTS_SIZE
NUM_EXAMPLES = 10000

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


def maybe_generate_data():
    '''Generate testing and training data if none exists in data_dir'''
    dest_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Log hook
    # TODO: not in use
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Generating %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    # generate training batches
    # constrained
    num_examples = NUM_EXAMPLES
    size = INPUTS_SIZE
    num_files = 2
    filenames = ['data_batch_%d.tfrecords' % i for i in range(1, num_files + 1)]

    for filename in filenames:
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            print('%s not found - generating...' % filename)
            utils.generate_constrained_dataset(filepath, _progress, **{
                'num_examples': num_examples,
                'size': size})
            print()
            statinfo = os.stat(filepath)
            print('Successfully generated', filename, statinfo.st_size, 'bytes.')

    # generate testing batches
    # random
    filename = 'test_batch.tfrecords'
    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        print('%s not found - generating...' % filename)
        # utils.generate_dataset(filepath, _progress, **{
        utils.generate_constrained_dataset(filepath, _progress, **{
            'num_examples': num_examples,
            'size': size})
        print()
        statinfo = os.stat(filepath)
        print('Successfully generated', filename, statinfo.st_size, 'bytes.')


class CABaseModel(object):
    '''
    CA model base class containing inputs streams and the shared static methods,
    - loss: loss function
    - optimizer: training op
    - prediction: prediction op

    Inference is implemented in each model inheriting this base class.
    It should contain,
    - inference: inference op, returning logits

    Models contains two separate input streams, one distorted input for training
    and one for testing.

    Notes:
        @staticmethod decorator marks a function as a regular function not inheriting its
        base class. See: http://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python
    '''
    def inference(self, inputs):
        assert 'Inference not implemented.'

    @staticmethod
    def loss(logits, labels):
        # loss function
        with tf.name_scope('loss'):
            # Compute moving averages of all individual losses and total loss
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy')
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

            # Add all losses to graph
            tf.add_to_collection('losses', cross_entropy_mean)
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss
    
    @staticmethod
    def optimizer(total_loss, global_step):
        # optimizer train_op

        # TRY: lr = tf.train.exponential_decay

        # Generate moving averages of all losses
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Compute gradients
        # Creates an average of all losses before applying minimize to optimizer op
        with tf.control_dependencies([loss_averages_op]):
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads = optimizer.compute_gradients(total_loss)  # minimize() 1/2

        # Apply gradients
        # minimize() 2/2
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Execute operations defined as arguments to control_dependencies before what's
        # inside its name scope.
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # optimizer = tf.train.AdamOptimizer(self._lr).minimize(loss)  # simple version of above
        return train_op

    @staticmethod
    def prediction(logits, labels, k):
        '''Evaluate the quality of the logits at predicting the label.
        
        Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size], with values in the
                range [0, NUM_CLASSES).

        Returns:
            A scalar int32 tensor with the number of examples (out of batch_size)
            that were predicted correctly.
        '''
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        with tf.name_scope('prediction'):
            # correct = tf.nn.in_top_k(logits, labels, k)
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            prediction = tf.reduce_mean(tf.cast(correct, tf.float32))
        # Return the number of true entries.
        return prediction

    @staticmethod
    def train_inputs():
        '''Distorted input stream for training.'''
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
        inputs, labels = input_pipeline.train_inputs(
            data_dir=data_dir,
            batch_size=FLAGS.batch_size)
        return inputs, labels

    @staticmethod
    def inputs(eval_data):
        '''Clean input stream for evaluation and testing.'''
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
        inputs, labels = input_pipeline.inputs(
            eval_data=eval_data,
            data_dir=data_dir,
            batch_size=FLAGS.batch_size)
        return inputs, labels


class CARNN(CABaseModel):
    '''
    CA RNN

    LSTM
    '''
    def inference(inputs):
        with tf.name_scope('inference'):

            # cell selection
            additional_cell_args = {}
            if FLAGS.cell_name == 'lstm':
                cell_fn = tf.contrib.rnn.LSTMCell
                # additional_cell_args.update({'state_is_tuple': True})
            elif FLAGS.cell_name == 'grid2lstm':
                cell_fn = tf.contrib.grid_rnn.Grid2LSTMCell
                additional_cell_args.update({'tied': True})
            elif FLAGS.cell_name == 'tf-gridlstm':
                cell_fn = tf.contrib.rnn.GridLSTMCell
                additional_cell_args.update({'state_is_tuple': True, 'num_frequency_blocks': [1]})
            else:
                raise Exception('Unsupported cell_name: {}'.format(FLAGS.cell_name))

            cell = cell_fn(FLAGS.state_size, **additional_cell_args)
            # initial_state = cell.zero_state(self._batch_size, tf.float32)

            # inputs
            # self._input_data, self._targets = self._input_pipeline()
            
            # with tf.device('/cpu:0'):
            # [batch, width, height, depth]
            # inputs = tf.reshape(self._input_data, [self._batch_size, self._rnn_size, 1])
            # inputs = tf.unstack(inputs, axis=1)

            # batch x width x height x depth
            # perm = [1, 2, 3, 0]
            # width x height x depth x batch
            # inputs = tf.transpose(self._input_data, perm)

            # batch x width
            inputs = tf.squeeze(inputs)

            layers = 10

            '''
            for g in range(1):
                for i in range(len(a)):
                    index = i + 2
                    n = a[max(0, index - 3):index]
                    print (n)
            '''

            # zero_states
            c_zero = tf.zeros([FLAGS.batch_size, FLAGS.state_size])
            h_zero = tf.zeros([FLAGS.batch_size, FLAGS.state_size])
            zero_state = tf.contrib.rnn.LSTMStateTuple(c_zero, h_zero)
            states = [zero_state] * FLAGS.rnn_size

            outputs = []
            inputs_as_list = tf.split(inputs, int(inputs.get_shape()[1]), axis=1)
            reuse = True
            with tf.variable_scope('lstm_grid') as scope:
                for l in range(layers):

                    for i, input_ in enumerate(inputs_as_list):
                        if reuse and i > 0:
                            scope.reuse_variables()

                        index = i + 2
                        neigh_states = states[max(0, index - 3): index]  # strided_slice

                        # concat neighouring states
                        # c_state = tf.concat([neigh_states[j].c for j in range(len(neigh_states))], 1)
                        # h_state = tf.concat([neigh_states[j].h for j in range(len(neigh_states))], 1)
                        # BUG: state_size grows linerarly per generation

                        # elementwise multiply neighbour states, essentially a conv
                        (c_states, h_states) = [list(a) for a in zip(*neigh_states)]
                        c_state = tf.reduce_sum(c_states, 0)
                        h_state = tf.reduce_sum(h_states, 0)
                        state = tf.contrib.rnn.LSTMStateTuple(c_state, h_state)

                        # print (state)
                        (output, new_state) = cell(input_, state, scope)
                        states[i] = new_state

                        if l == layers - 1:  # last generation
                            outputs.append(output)

            # rnn
            with tf.variable_scope('rnn'):
                # V matrices
                softmax_w = tf.get_variable('softmax_w', [FLAGS.rnn_size * FLAGS.state_size, FLAGS.num_classes])
                softmax_b = tf.get_variable('softmax_b', [FLAGS.num_classes])

            with tf.name_scope('softmax'):
                output = tf.concat(outputs, 1)
                logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        return logits
    

class CAConv(CABaseModel):
    @staticmethod
    def inference(inputs):
        '''Builds inference for the graph.

        input size: batch x width x height x depth

        Args:
            inputs: inputs returned from train_inputs() or inputs()

        Returns:
            logits
        '''
        # Group scoped ops together in graph
        with tf.name_scope('inference'):

            # TRY: for optimization: weight decay

            dtype = tf.float32
            initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=dtype)
            activation = tf.nn.relu

            # Input convolution layer
            with tf.variable_scope('conv1') as scope:
                # Increase input depth from 1 to state_size
                kernel = tf.get_variable('weights', shape=[3, 1, 1, FLAGS.state_size], 
                                         initializer=initializer, dtype=dtype)
                biases = tf.get_variable('biases', shape=[FLAGS.state_size], 
                                         initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv1 = activation(pre_activation, name=scope.name)

            # Cellular Automaton module
            with tf.variable_scope('ca_conv') as scope:
                # List of all layer states
                state_layers = [conv1]
                for layer in range(FLAGS.num_layers):
                    if FLAGS.reuse and layer > 0:
                        # Mark variables to be reused within the same name scope
                        # after layer 0
                        scope.reuse_variables()
                    # layer input is state_size to state_size
                    kernel = tf.get_variable('weights', shape=[3, 1, FLAGS.state_size, FLAGS.state_size], 
                                             initializer=initializer, dtype=dtype)
                    biases = tf.get_variable('biases', shape=[FLAGS.state_size], 
                                             initializer=tf.constant_initializer(0.0))
                    # Previous layer as input
                    conv = tf.nn.conv2d(state_layers[-1], kernel, strides=[1, 1, 1, 1], padding='SAME')
                    pre_activation = tf.nn.bias_add(conv, biases)
                    conv_state = activation(pre_activation, name=scope.name)
                    state_layers.append(conv_state)

            # Output module
            with tf.variable_scope('output_conv'):
                # reduce depth from state_size to 1
                kernel = tf.get_variable('weights', shape=[3, 1, FLAGS.state_size, 1], 
                                         initializer=tf.truncated_normal_initializer(
                                             stddev=1.0 / tf.square(float(FLAGS.state_size)), 
                                             dtype=dtype), 
                                         dtype=dtype)
                biases = tf.get_variable('biases', shape=[1], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(conv_state, kernel, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                output = activation(pre_activation, name=scope.name)

            # Flatten output layer for classification
            # tf.sparse_softmax_cross_entropy_with_logits loss function applies softmax internally
            with tf.variable_scope('softmax_linear'):
                # flatten to one dim
                reshape = tf.reshape(output, [FLAGS.batch_size, -1])
                softmax_w = tf.get_variable('softmax_w', shape=[9, FLAGS.num_classes])
                softmax_b = tf.get_variable('softmax_b', shape=[FLAGS.num_classes])
                softmax_linear = tf.nn.xw_plus_b(reshape, softmax_w, softmax_b)
                logits = softmax_linear

            return logits

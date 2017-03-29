import tensorflow as tf

import input_pipeline
import os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            '''Size of batch.''')
tf.app.flags.DEFINE_integer('state_size', 128,
                            '''Size of cell states.''')
tf.app.flags.DEFINE_integer('rnn_size', 5,
                            '''Size of rnn.''')
tf.app.flags.DEFINE_integer('learning_rate', 0.01,
                            '''Learning rate.''')
tf.app.flags.DEFINE_integer('num_classes', 2,
                            '''Number of label classes.''')
tf.app.flags.DEFINE_string('data_dir', 'tmp/data',
                           '''Path to the data directory.''')
tf.app.flags.DEFINE_string('cell_name', 'lstm',
                           '''Name of cell to use.''')

'''
version 0.1
    + added ca in 1d
    + moved inputs to separate file to use with both train and eval
    - removed model class in favour to scrips
    - removed @properties in favour of external creation (less complexity)
'''
    

def inference(boards):
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
        inputs = tf.squeeze(boards)

        generations = 10

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
            for g in range(generations):

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

                    if g == generations - 1:  # last generation
                        outputs.append(output)

        # rnn
        with tf.variable_scope('rnn'):
            # V matrixes
            softmax_w = tf.get_variable('softmax_w', [FLAGS.rnn_size * FLAGS.state_size, FLAGS.num_classes])
            softmax_b = tf.get_variable('softmax_b', [FLAGS.num_classes])

        with tf.name_scope('softmax'):
            output = tf.concat(outputs, 1)
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    return logits
    

def loss(logits, labels):
    # loss function
    with tf.name_scope('loss'):
        # Compute moving averages of all individual losses and total loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
    

def optimizer(total_loss, global_step):
    # optimizer train_op

    # TRY: lr = tf.train.exponential_decay

    # Generate moving averages of all losses
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads = optimizer.compute_gradients(total_loss)  # minimize() 1/2

    # Apply gradients
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)  # minimize() 2/2

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # optimizer = tf.train.AdamOptimizer(self._lr).minimize(loss)  # simple version of above
    return train_op


def prediction(logits, labels, k):
    # evaluation
    with tf.name_scope('prediction'):
        correct = tf.nn.in_top_k(logits, labels, k)
        prediction = tf.reduce_mean(tf.cast(correct, tf.float32))
        return prediction


def train_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
    boards, labels = input_pipeline.train_inputs(
        data_dir=data_dir,
        batch_size=FLAGS.batch_size)
    return boards, labels


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'batches-bin')
    boards, labels = input_pipeline.inputs(
        eval_data=eval_data,
        data_dir=data_dir,
        batch_size=FLAGS.batch_size)
    return boards, labels

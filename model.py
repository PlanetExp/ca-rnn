import tensorflow as tf
import numpy as np


class ExtendedGRUcell(tf.contrib.rnn.GRUCell):
    def __call__(self, inputs, state, scope=None):
        new_h, new_h = super().__call__(inputs, state, scope=None)
        
        print('extending')
        
        return new_h, new_h
        
        
class Model(object):
    def __init__(self,
                 batch_size,
                 state_size,
                 num_classes,
                 rnn_size,
                 learning_rate,
                 cell_name):
        self._cell_name = cell_name
        self._batch_size = batch_size
        self._state_size = state_size
        self._num_classes = num_classes
        self._rnn_size = rnn_size
        self._lr = learning_rate
        
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        self._create_prediction()
    
    @property
    def inference(self):
        return self._inference
        
    @property
    def loss(self):
        return self._loss
    
    @property
    def optimizer(self):
        return self._optimizer
        
    @property
    def prediction(self):
        return self._prediction
    
    def _create_inference(self):
        with tf.name_scope('inference'):

            # cell selection
            additional_cell_args = {}
            if self._cell_name == 'lstm':
                cell_fn = tf.contrib.rnn.LSTMCell
                # additional_cell_args.update({'state_is_tuple': True})
            elif self._cell_name == 'grid2lstm':
                cell_fn = tf.contrib.grid_rnn.Grid2LSTMCell
                additional_cell_args.update({'tied': True})
            elif self._cell_name == 'tf-gridlstm':
                cell_fn = tf.contrib.rnn.GridLSTMCell
                additional_cell_args.update({'state_is_tuple': True, 'num_frequency_blocks': [1]})
            else:
                raise Exception('Unsupported cell_name: {}'.format(self._cell_name))

            cell = cell_fn(self._state_size, **additional_cell_args)
            # initial_state = cell.zero_state(self._batch_size, tf.float32)

            # inputs
            self._input_data, self._targets = self._input_pipeline()
            
            # with tf.device('/cpu:0'):
            # [batch, width, height, depth]
            # inputs = tf.reshape(self._input_data, [self._batch_size, self._rnn_size, 1])
            # inputs = tf.unstack(inputs, axis=1)

            # batch x width x height x depth
            # perm = [1, 2, 3, 0]
            # width x height x depth x batch
            # inputs = tf.transpose(self._input_data, perm)

            # batch x width
            inputs = self._input_data
            inputs = tf.squeeze(inputs)

            generations = 3

            '''
            for g in range(1):
                for i in range(len(a)):
                    index = i + 2
                    n = a[max(0, index - 3):index]
                    print (n)
            '''

            # zero_states
            c_zero = tf.zeros([self._batch_size, self._state_size])
            h_zero = tf.zeros([self._batch_size, self._state_size])
            zero_state = tf.contrib.rnn.LSTMStateTuple(c_zero, h_zero)
            states = [zero_state] * self._rnn_size

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

                        # pad with zero or maybe clamp sides?
                        if (len(neigh_states) < 3):
                            neigh_states.append(zero_state)
                        # TODO: left/right

                        # concat neighouring states
                        # c_state = tf.concat([neigh_states[j].c for j in range(len(neigh_states))], 1)
                        # h_state = tf.concat([neigh_states[j].h for j in range(len(neigh_states))], 1)
                        # TODO: size grows per generation

                        # elementwise multiply neighbour states, essentially a conv
                        c_state = neigh_states[2].c
                        h_state = neigh_states[2].h
                        for m in range(len(neigh_states) - 1):
                            c_state *= neigh_states[m].c
                            h_state *= neigh_states[m].h
                        state = tf.contrib.rnn.LSTMStateTuple(c_state, h_state)

                        # print (state)
                        (output, new_state) = cell(input_, state, scope)
                        states[i] = new_state

                        if g == generations - 1:  # last generation
                            outputs.append(output)

            # rnn
            with tf.variable_scope('rnn'):
                # V matrixes
                softmax_w = tf.get_variable('softmax_w', [self._rnn_size * self._state_size, self._num_classes])
                softmax_b = tf.get_variable('softmax_b', [self._num_classes])

                # inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size]
                # outputs, final_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=initial_state)

            with tf.name_scope('softmax'):
                output = tf.concat(outputs, 1)
                self._logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        
    def _create_loss(self):
        # loss function
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self._targets, name='cross_entropy')
            self._loss = tf.reduce_mean(cross_entropy)
            
    def _create_optimizer(self):
        # optimizer
            self._optimizer = tf.train.AdamOptimizer(self._lr).minimize(self._loss)
            
    def _create_prediction(self):
        # evaluation
        with tf.name_scope('prediction'):
            correct = tf.nn.in_top_k(self._logits, self._targets, 1)
            self._prediction = tf.reduce_mean(tf.cast(correct, tf.float32))

    def _input_pipeline(self):
        filename_queue = tf.train.string_input_producer(
            ['data/const_train_1_200000x5x1x1.tfrecords',
             'data/const_train_2_200000x5x1x1.tfrecords'], num_epochs=None)

        def read_and_decode(filename_queue):
            # read
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            feature_map = {
                'x': tf.FixedLenFeature(
                    shape=[], dtype=tf.string),
                'y': tf.FixedLenFeature(
                    shape=[], dtype=tf.int64,
                    default_value=None)
            }
            parsed = tf.parse_single_example(serialized_example, feature_map)

            # decode
            width = 5
            height = 1
            depth = 1

            features = tf.decode_raw(parsed['x'], tf.int64)
            features = tf.reshape(features, [width, height, depth])
            features = tf.cast(features, dtype=tf.float32)
            labels = parsed['y']

            return features, labels

        features, labels = read_and_decode(filename_queue)

        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 + self._batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [features, labels],
            batch_size=self._batch_size,
            capacity=capacity,
            num_threads=4,
            allow_smaller_final_batch=False,
            min_after_dequeue=min_after_dequeue)

        return example_batch, label_batch

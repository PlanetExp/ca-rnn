'''Various utilities

Created 6 mars 2017
@author Frederick Heidrich
'''
import numpy as np
import tensorflow as tf

from collections import namedtuple
from dataset1d import Dataset1d

import sys
import os


def get_connection_length(board, start_set=None, target_set=None):
    '''
        Determine the connectivity between the cell sets 'start_set' and
        'target_set'. That is, check for the existence of a path of connected stones
        from start_set to target_set.

        If start_set is None, it is initialized with the set of stones of row 0 (top row).
        if target_set is None, it is initialized with the set of stones of the bottom row.


        Return  connection_length
             connection_length is None  is the cell sets are not connected
             otherwise, it is the length of the connecting path
             
        Params
            board : uint8 np array,
                    zero -> background
                    non-zero -> stone
            start_set, target_set : cell coords  of the cell sets.sys.maxsize
                                    Lists of (row,col) pairs
                                    
    '''
    num_rows, num_cols = board.shape[0], board.shape[1]

    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
    
    def getNeighbours(A, T):
        '''
            Return the set 'N' of stony neighbors of 'A', that are not in 'T' (taboo set)
            The neighbors are represented with the pairs of their (row, column) coordinates.
            Note that this is a sub-function.
        '''
        N = set()
        for r, c in A:
            for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                rn, cn = r + dr, c + dc
                if ((0 <= rn < num_rows) and (0 <= cn < num_cols) and board[rn, cn] != 0 and (rn, cn) not in T):
                    N.add((rn, cn))
        return N
    
    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .

    if start_set is None:
        start_set = [(0, c) for c in range(num_cols) if board[0, c] != 0]

    if target_set is None:
        target_set = [(num_rows - 1, c) for c in range(num_cols) if board[num_cols - 1, c] != 0]

    g = 0  # generation index
    T = set()  # taboo cells
    F = set(start_set)  # frontier

    while not F.intersection(target_set):  # reached target?
        g += 1
        T.update(F)
        F = getNeighbours(F, T)
        if not F:  # cul de sac!
            return None
    return g


def board_generator(
        shape,
        min_connection_length,
        max_connection_length,
        k_value=2,
        stone_probability=0.5):
    fail_count = 0
    while True:
        # Generate boards with random distribution of stones
        board = np.random.choice(k_value, 
                                 size=[shape[0], shape[1], shape[2]],
                                 p=[1 - stone_probability, stone_probability])

        connection_length = get_connection_length(board)

        if connection_length is None:
            # connectivity = sys.maxsize
            connectivity = 0
        else:
            # connectivity = connection_length
            connectivity = 1
        
        if (min_connection_length <= connectivity <= max_connection_length):
            # print ("success after ", fail_count, " failures")
            fail_count = 0  # reset counter
            yield board, connectivity  # X, y
        else:
            fail_count += 1  # print "failed"


def generate_constrained_dataset(
        # filename,
        # size,
        # num_positive_examples,
        # num_negative_examples,
        # stone_probability=0.5,
        # k_value=2,
        # verbose=False):
        filepath,
        progress_fn=None,
        shape=None,
        num_examples=None,
        shuffle=True,
        stone_probability=0.5):
    '''
    Generates a constrained dataset of 50/50 positive and negative examples
        and converts them to a .tfrecords file on the disk where filepath
        is specified.

    Args:
        filepath: path to file to be written
        progress_fn: progress logger hook (unused)
        shape: shape of the data to be written
        num_examples: number of examples to generate, half of which will be positive
        shuffle: whether or not to shuffle this dataset before writing to file (True)
        stone_probability: probability distribution of 'black' stones on the generated boards
    '''

    width = shape[0]
    height = shape[1]
    depth = shape[2]

    # theoretical_max_length = int(height * width / 2 + width / 2)
    inputs = np.empty((num_examples, width, height, depth), np.int8)  # boards
    labels = np.empty((num_examples, ), np.int8)  # connection length

    pos_board_generator = board_generator(shape, 1, 1, stone_probability=stone_probability)
    neg_board_generator = board_generator(shape, 0, 0, stone_probability=stone_probability)

    num_pos_examples = num_examples // 2

    for i in range(num_examples):
        if i < num_pos_examples:
            inputs[i], labels[i] = next(pos_board_generator)
        else:
            inputs[i], labels[i] = next(neg_board_generator)

        if i % 1000 == 0:
            print('Generating {} boards. Progress: {}/{}'.format(
                'positive' if i < num_pos_examples else 'negative', i, num_examples))

    # ad-hoc shuffling
    # NOTE: creates copies
    if shuffle:
        perm = np.random.permutation(len(inputs))
        inputs = inputs[perm]
        labels = labels[perm]

    # save to file
    _convert_to_tfrecords(inputs, shape, labels, filepath)


def _bytes_feature(values):
    '''Helper function to generate an tf.train.Feature protobuf'''
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(values):
    '''Helper function to generate an tf.train.Feature protobuf'''
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _data_to_tfexample(encoded_image, height, width, class_id):
    '''Helper function to generate an tf.train.Example protobuf'''
    return tf.train.Example(features=tf.train.Features(
        feature={
            'image/encoded': _bytes_feature(encoded_image)
            # 'image/class/label': _int64_feature(class_id),
            # 'image/height': _int64_feature(height),
            # 'image/uint8image': _int64_feature(uint8image)
        }))


def _convert_to_tfrecords(inputs, shape, labels, filepath):
    '''Helper function to write tfrecords file

    Args:
        inpupts:
        shape:
        labels:
        filepath:
    '''
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    
    inputs = inputs.astype(np.int8)
    labels = labels.astype(np.int8)

    for i in range(len(inputs)):
        features = inputs[i].tobytes()
        label = labels[i].tobytes()

        # Encode label byte as part of the record
        encoded_image = b''.join([label, features])

        example = _data_to_tfexample(encoded_image, shape[0], shape[1], label)
        # size = example.BytesSize()
        writer.write(example.SerializeToString())
    writer.close()

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
        board = np.random.choice(k_value, size=[shape[0], shape[1], shape[2]], p=[1 - stone_probability, stone_probability])

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
        num_examples=None,
        size=None):

    width = size[0]
    height = size[1]
    depth = size[2]

    # theoretical_max_length = int(height * width / 2 + width / 2)
    inputs = np.empty((num_examples, width, height, depth), np.int)  # boards
    labels = np.empty((num_examples, ), np.int)  # connection length

    pos_board_generator = board_generator(size, 1, 1)
    neg_board_generator = board_generator(size, 0, 0)

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
    perm = np.random.permutation(len(inputs))
    inputs = inputs[perm]
    labels = labels[perm]

    # save to file
    convert_to_tfrecords(inputs, labels, filepath)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecords(x, y, filepath):
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    
    for i in range(len(x)):
        features = x[i].tobytes()
        label = int(y[i])
        
        # Example proto
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'x': _bytes_feature(features),
                    'y': _int64_feature(label)
                }))
        writer.write(example.SerializeToString())
    writer.close()


# def build_1d_dataset(
#         width=8,
#         depth=1,
#         n_samples=100,
#         k_value=2,
#         train_split=0.8,
#         valid_split=0.5,
#         verbose=False):
    
#     x = np.random.randint(0, k_value, size=[n_samples, width, depth])
#     y = np.zeros(n_samples, dtype=int)
    
#     # samples, [width, depth]
#     for i, board in enumerate(x):
#         # count connection length
#         connection_length = 0
#         # width, depth
#         for j, grid in enumerate(board):
#             if grid == [1]:
#                 connection_length += 1
#             else:
#                 break
                
#         if connection_length == width:
#             y[i] = 1
#         else:
#             y[i] = 0
# #         y[i] = connection_length

#     dataset = namedtuple('Dataset', ['train', 'valid', 'test'])
    
#     # Split dataset
#     n_train = int(n_samples * train_split)
#     n_valid = int((n_samples - n_train) * valid_split)
    
#     dataset.train = Dataset1d(x[:n_train], y[:n_train])
#     dataset.valid = Dataset1d(x[:n_valid], y[:n_valid])
#     dataset.test = Dataset1d(x[:n_valid], y[:n_valid])
    
#     return dataset

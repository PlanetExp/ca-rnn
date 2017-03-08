'''Various utilities

Created 6 mars 2017
@author Frederick Heidrich
'''
import numpy as np

from collections import namedtuple
from dataset1d import Dataset1d


def build_1d_dataset(
        width=8,
        depth=1,
        n_samples=100,
        k_value=2,
        train_split=0.8,
        valid_split=0.5,
        verbose=False):
    
    x = np.random.randint(0, k_value, size=[n_samples, width, depth])
    y = np.zeros(n_samples, dtype=int)
    
    # samples, [width, depth]
    for i, board in enumerate(x):
        # count connection length
        connection_length = 0
        # width, depth
        for j, grid in enumerate(board):
            if grid == [1]:
                connection_length += 1
            else:
                break
                
        if connection_length == width:
            y[i] = 1
        else:
            y[i] = 0
#         y[i] = connection_length

    dataset = namedtuple('Dataset', ['train', 'valid', 'test'])
    
    # Split dataset
    n_train = int(n_samples * train_split)
    n_valid = int((n_samples - n_train) * valid_split)
    
    dataset.train = Dataset1d(x[:n_train], y[:n_train])
    dataset.valid = Dataset1d(x[:n_valid], y[:n_valid])
    dataset.test = Dataset1d(x[:n_valid], y[:n_valid])
    
    return dataset

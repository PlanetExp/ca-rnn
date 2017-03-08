'''
Created 6 mars 2017
@author: Frederick Heidrich
'''

import numpy as np


class Dataset1d(object):
    def __init__(self, x, y):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._n_samples = x.shape[0]
        self._x = x
        self._y = y
        
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._n_samples)
            np.random.shuffle(perm0)
            self._x = self._x[perm0]
            self._y = self._y[perm0]
            
        if start + batch_size > self._n_samples:
            # finished
            self._epochs_completed += 1
            rest_samples = self._n_samples - start
            x_rest_part = self._x[start:self._n_samples]
            y_rest_part = self._y[start:self._n_samples]
            
            if shuffle:
                perm = np.arange(self._n_samples)
                np.random.shuffle(perm)
                self._x = self._x[perm]
                self._y = self._y[perm]

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size - rest_samples
            end = self._index_in_epoch
            x_new_part = self._x[start:end]
            y_new_part = self._y[start:end]
            
            return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate((y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            
            return self._x[start:end], self._y[start:end]
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def n_samples(self):
        return self._n_samples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

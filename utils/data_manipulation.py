from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys

def shuffle_data(X, y, seed=None):
    """
    Random shuffle of the samples in X (feature set) and y(target set)
    X: feature set
    y: target set
    seed: seed for numpy random
    
    return -> the shuffled 
    """

    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def batch_iterator(X, y=None, batch_size=64):
    """
    Batch generator to generate batches from the dataset
    X: feature set
    y: targetset
    batch_size: default=64
    """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]



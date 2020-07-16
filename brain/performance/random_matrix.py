from brain.performance.multithreaded_rng import *

import numpy as np

# TODO: seems unused, remove
class RandomMatrix(object):
    def __init__(self, n, m, p, mrng=None):
        self.mrng = mrng or MultithreadedRNG()
        self.values = np.empty((n, m))
        self.mrng.fill(self.values, p)

    def __setitem__(self, key, value):
        self.values[key] = value
    
    def __getitem__(self, key):
        return self.values[key]
    
    def sum_rows(self):
        return np.sum(self.values, axis=0)



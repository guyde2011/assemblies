import numpy as np

from brain.performance.random_matrix import RandomMatrix

# TODO: this file seems unused, remove


def expectation(a):
    return np.sum(a)/a.size


def cov(a, b):
    return expectation(a*b) - expectation(a)*expectation(b)


if __name__ == '__main__':
    arr = RandomMatrix().multi_generate(500, 500, 0.4)
    # Should be small
    print(cov(arr[0:200], arr[200:400]))

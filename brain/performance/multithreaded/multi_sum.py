import numpy as np

from brain.performance.multithreaded.multithreaded import multithreaded


@multithreaded
def multi_sum(arr, lst):
    return np.sum(arr[lst, :], axis=0)


@multi_sum.params
def multi_sum_params(threads, arr, indices):
    step = np.ceil(len(indices) / threads).astype(np.int_)
    return (((arr, indices[i*step:min((i+1)*step, len(indices))]), {}) for i in range(threads))


@multi_sum.after
def multi_sum_after(out):
    return sum(out)


if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 4]])

    print(multi_sum(a))

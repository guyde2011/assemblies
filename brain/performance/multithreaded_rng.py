from numpy.random import Generator, PCG64
import multiprocessing
import concurrent.futures
import numpy as np
from brain.performance.multithreaded import multithreaded

from typing import Optional


class MultithreadedRNG:
    def __init__(self, threads=None, seed=None):
        rg = PCG64(seed)
        self._random_generators = [rg]
        last_rg = rg
        self.multi_generate = multithreaded(self._multi_generate, threads=threads)
        self.multi_generate.after(self._multi_generate_after)
        self.multi_generate.params(self._multi_params)
        for _ in range(len(self.multi_generate)-1):
            new_rg = last_rg.jumped()
            self._random_generators.append(new_rg)
            last_rg = new_rg
        self._random_generators = [Generator(rg) for rg in self._random_generators]

    @staticmethod
    def _multi_generate(rg: Generator, out: np.array, first: int, last: int, p: float):
        out[first:last] = rg.binomial(1, p, out[first:last].shape)
        return out

    def _multi_params(self, threads: int, height: int, width: int, prob: float):
        step = np.ceil(height / threads).astype(np.int_)
        out = np.empty((height, width))
        return (((self._random_generators[i], out, i * step, (i + 1) * step, prob), {}) for i in range(threads))

    @staticmethod
    def _multi_generate_after(outs):
        return outs[0] if outs else np.empty((0, 0), dtype='float64')


if __name__ == '__main__':
    n = 10000
    a = MultithreadedRNG().multi_generate(n, n, 0.1)
    print(np.sum(a) / n**2)

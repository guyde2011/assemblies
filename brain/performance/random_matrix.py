from numpy.random import Generator, PCG64
import numpy as np
from brain.performance.multithreaded import multithreaded


class RandomMatrix:
    """
    A class which generates a random matrix using multithreading.

    After creating an instance of the class, use multi_generate(height, width, p) to create an n x m matrix filled with
    i.i.d. bernoulli(p) random variables.

    """

    def __init__(self, threads: int = None, seed=None):
        """
        :param threads: Number of threads. no thread will use the number of cpus.
        :param seed: an optional seed for the prng.
        """
        rng = PCG64(seed)
        self._random_generators = [rng]

        self.multi_generate = multithreaded(self._multi_generate, threads=threads)
        self.multi_generate.after(self._multi_generate_after)
        self.multi_generate.params(self._multi_params)

        for _ in range(len(self.multi_generate) - 1):
            # Each prng will be used by a different thread, so we need to have different prngs.
            rng = rng.jumped()
            self._random_generators.append(rng)
        self._random_generators = [Generator(rng) for rng in self._random_generators]

    @staticmethod
    def _multi_generate(rng: Generator, out: np.array, first: int, last: int, prob: float):
        """
        Every thread generates out[first:last].
        :param rng: The rng used by the function.
        :param out: the output array.
        :param first: first row.
        :param last: last row.
        :param prob: Bernoulli parameter.
        :return:
        """
        out[first:last] = rng.binomial(1, prob, out[first:last].shape)
        return out

    def _multi_params(self, threads: int, height: int, width: int, prob: float):
        """
        Handling the parameters to multi_generate.
        :param threads: number of threads
        :param height: height of matrix
        :param width: width of matrix
        :param prob: Bernoulli parameter.
        :return:
        """
        step = np.ceil(height / threads).astype(np.int_)
        out = np.empty((height, width))
        return (((self._random_generators[i], out, i * step, (i + 1) * step, prob), {}) for i in range(threads))

    @staticmethod
    def _multi_generate_after(outs):
        return outs[0] if outs else np.empty((0, 0), dtype='float64')


# TODO: this looks like a nice base to extract as a test! :)
if __name__ == '__main__':
    n = 10000
    a = RandomMatrix().multi_generate(n, n, 0.1)
    print(np.sum(a) / n ** 2)

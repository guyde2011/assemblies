import multiprocessing
import concurrent.futures

from typing import Callable, Any
import functools


def __identity__(x):
    return x


class Multithreaded:
    # TODO: document. especially the interface and the way this class can be used
    def __init__(self, func, threads):
        self._function = func
        self._params: Callable[[int, ...], Any] = lambda _, *args, **kwargs: (args, kwargs)
        self._after = __identity__
        self.__name__ = func.__name__

        if hasattr(func, '__docs__'):
            self.__docs__ = func.__docs__
        if hasattr(func, '__signature__'):
            self.__signature__ = func.__signature__

        self._threads = threads or multiprocessing.cpu_count()
        self._executor = concurrent.futures.ThreadPoolExecutor(self._threads)

    def params(self, params):
        self._params = params

    def after(self, func):
        self._after = func

    def __call__(self, *args, **kwargs):
        futures = {}
        params = self._params(self._threads, *args, **kwargs)
        outs = [None] * self._threads
        for i, value in enumerate(params):
            t_args, t_kwargs = value

            def do_thread(m_i, m_args, m_kwargs):
                outs[m_i] = self._function(*m_args, **m_kwargs)

            futures[self._executor.submit(do_thread, i, t_args, t_kwargs)] = i
        concurrent.futures.wait(futures)
        return self._after(outs)

    def __get__(self, instance, owner):
        if instance:
            mt = Multithreaded(self._function.__get__(instance, owner), self._threads)
            mt.after(self._after.__get__(instance, owner))
            mt.params(self._params.__get__(instance, owner))
            return mt

    def __len__(self):
        return self._threads


# TODO: this should be a static method instead `Multithreaded` class - it is only related to that class
# TODO 2: why is the * parameter necessary
def multithreaded(func=None, *, threads=None):
    return Multithreaded(func, threads) if func else (lambda f: Multithreaded(f, threads))

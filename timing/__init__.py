import profile as prof
import time
import functools
import inspect
import click
import traceback
import sys


from typing import Optional, Callable, Any


def run_timed(func: Callable, name: Optional[str] = None, *args, **kwargs):
    name = name or func.__name__
    t0 = time.time()
    try:
        ret = func(*args, **kwargs)
    except BaseException as e:
        click.echo(f"\x1b[31m[ERROR] Function \x1b[1m{name}\x1b[0m\x1b[31m failed with error \n\t> {e}\x1b[37m")
        traceback.print_tb(sys.exc_info()[2])
        return None
    t1 = time.time()
    click.echo(f'\x1b[33m[INFO] Executed \x1b[1m{name}\x1b[0m\x1b[33m in {int((t1 - t0) * 1000)}ms\x1b[37m')
    return ret


def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        if inspect.isgeneratorfunction(func):
            gen = func(*args, **kwargs)
            t0 = time.time()

            @functools.wraps(func)
            def run_func():
                try:
                    return next(gen)
                except StopIteration:
                    return None
            ret = next(gen)
            while ret is not None:
                ret = run_timed(run_func, f'{func.__name__}:{ret}')
            t1 = time.time()
            click.echo(f'\x1b[33m[INFO] Executed \x1b[1m{func.__name__}\x1b[0m\x1b[33m'
                       f' in {int((t1 - t0) * 1000)}ms\x1b[37m')
        else:
            run_timed(func, func.__name__, *args, **kwargs)

    return wrapper


def profiling(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return prof.runctx('func(*args, **kwargs)',
                           globals=globals(), locals={'func': func, 'args': args, 'kwargs': kwargs})

    return wrapper


def profile(func):
    func.__profile__ = True
    return func

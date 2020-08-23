from functools import wraps
from inspect import Parameter

from utils.argument_manipulation import argument_restrict, argument_extend

# TODO: remove unused

class Repeat:
    """
    Decorator for repeating a function
    """

    def __init__(self, default: int = 1, resolve=None):
        """
        Initialize a repeat decorator
        :param default: Default amount of times to repeat function
        :param resolve: Method in which to resolve how many to repeats to perform
        """
        self.default = default
        self.resolve = resolve

    def __call__(self, func):
        restricted_func = argument_restrict(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            t = self.resolve(*args, **kwargs) if self.resolve else self.default

            result = None
            for _ in range(t):
                result = restricted_func(*args, **kwargs)

            return result

        return argument_extend(Parameter('t', Parameter.KEYWORD_ONLY, default=None, annotation=int),
                               restrict=False)(wrapper)

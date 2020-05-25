from inspect import signature, Parameter
from functools import wraps


def argument_restrict(func):
    """Argument restriction decorator, removes unwanted arguments"""
    sig = signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **{name: value for name, value in kwargs.items() if name in sig.parameters})

    return wrapper


def argument_extend(*params: Parameter, restrict: bool = True):
    """Argument extender decorator, add arguments to signature"""
    def decorate(func):
        func = argument_restrict(func) if restrict else func
        sig = signature(func)
        new_params = list(sig.parameters.values()) \
                     + [param for param in params if param.name not in sig.parameters]
        new_sig = sig.replace(parameters=new_params)
        func.__signature__ = new_sig
        return func

    return decorate

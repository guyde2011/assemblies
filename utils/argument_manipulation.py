from inspect import signature as _signature
from inspect import Parameter
from functools import wraps


def signature(func, use_original=False):
    """Better signature function"""
    return _signature(func) if use_original else (getattr(func, '__signature__', None) or _signature(func))


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


def argument_explicit_restrict(*params: str):
    """Removes arguments from function signature"""
    def decorate(func):
        sig = signature(func)
        new_params = list(param for name, param in sig.parameters.items() if name not in params)
        new_sig = sig.replace(parameters=new_params)
        func.__signature__ = new_sig
        return func

    return decorate

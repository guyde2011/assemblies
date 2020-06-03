from functools import wraps
from inspect import Parameter
from typing import Tuple

from utils.argument_manipulation import argument_extend, argument_explicit_restrict, signature
from utils.bindable import Bindable, protected_bindable
from utils.blueprints.recording import Recording
from utils.class_manipulation import variables, FunctionWrapper


class Recordable:
    def __init__(self, *functions: str):
        self.functions: Tuple[str] = functions

    @staticmethod
    def wrap_method(func):
        class Wrapper(FunctionWrapper):
            def __init__(self):
                super(Wrapper, self).__init__(signature(Wrapper.wrapped(func)))
            
            @staticmethod
            def wrapped(function):
                @wraps(function)
                def wrapper(*args, recording: Recording = None, **kwargs):
                    if recording is not None:
                        recording.append(function, args, kwargs)
                    else:
                        return function(*args, **kwargs)

                return argument_extend(
                    Parameter('recording', Parameter.KEYWORD_ONLY, default=None, annotation=Recording),
                    restrict=False)(wrapper)

            def __get__(self, instance, owner):
                return Wrapper.wrapped(func.__get__(instance, owner))

        return Wrapper()

    @staticmethod
    def wrap_class(cls, functions: Tuple[str, ...]):
        for func_name, (func, bound_func) in variables(cls).items():
            if callable(bound_func) and func_name in functions:
                setattr(cls, func_name, protected_bindable(Recordable.wrap_method(func)))

        cls = Bindable.wrap_class(cls, ('recording', ))

        for func_name, (func, bound_func) in variables(cls).items():
            if callable(bound_func) and func_name in functions:
                class Wrapper(FunctionWrapper):
                    def __init__(self, function):
                        super(Wrapper, self).__init__(signature(argument_explicit_restrict('recording')(function)))
                        self.function = function

                    def __get__(self, instance, owner):
                        return argument_explicit_restrict('recording')(self.function.__get__(instance, owner))

                setattr(cls, func_name, Wrapper(func))

        return cls

    def __call__(self, cls):
        return Recordable.wrap_class(cls, self.functions)

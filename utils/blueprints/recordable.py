from functools import wraps
from inspect import Parameter
from typing import Tuple, Union, Dict, Optional

from utils.argument_manipulation import argument_extend, argument_explicit_restrict, signature
from utils.bindable import Bindable, protected_bindable
from utils.blueprints.recording import Recording
from utils.class_manipulation import variables, FunctionWrapper
from utils.implicit_resolution import ImplicitResolution


class Recordable:
    """
    Recordable decorator for classes
    This enables recoding actions which will then be played back later,
     with some additional parameters
    """

    def __init__(self, *functions: Union[str, Tuple[str, bool]], resolution: Optional[ImplicitResolution] = None):
        """
        Create a recordable decorator
        :param functions: A list of strings or tuples of functions to record, boolean parameter of the tuple allows
                           stating to execute the function as well as recording it
        :param resolution: ImplicitResolution providing a way to resolve the recording object to record to
        """
        self.resolution: Optional[ImplicitResolution] = resolution
        self.functions: Dict[str, bool] = {}

        for function in functions:
            if isinstance(function, str):
                # Default behaviour is to record function without executing
                self.functions[function] = False
            else:
                function_name, value = function
                self.functions[function_name] = value

    @staticmethod
    def wrap_method(func, execute_anyway: bool, resolution: Optional[ImplicitResolution]):
        """
        Wrap a method to comply with recording
        :param func: Function to decorate
        :param execute_anyway: Boolean flag on whether to execute function even if it should be recorded
        :param resolution: ImplicitResolution providing a way to resolve the recording object to record to
        :return: Decorated method
        """

        class Wrapper(FunctionWrapper):
            """
            Wrapper class to support recording
            """

            def __init__(self):
                super(Wrapper, self).__init__(signature(Wrapper.wrapped(func)))

            @staticmethod
            def wrapped(function):
                @wraps(function)
                def wrapper(*args, recording: Recording = None, **kwargs):
                    if recording is not None:
                        # If recording parameter was provided, record function
                        recording.append(function, args, kwargs)
                        if not execute_anyway:
                            # Check if function should be executed anyway
                            return

                    return function(*args, **kwargs)

                # Add a recording parameter to the signature
                return argument_extend(
                    Parameter('recording', Parameter.KEYWORD_ONLY, default=None, annotation=Recording),
                    restrict=False)(wrapper)

            def __get__(self, instance, owner):
                return Wrapper.wrapped(func.__get__(instance, owner))

        return resolution(Wrapper()) if resolution else Wrapper()

    @staticmethod
    def wrap_class(cls, functions: Dict[str, bool], resolution: Optional[ImplicitResolution]):
        """
        Wrap a class to comply with recording
        :param cls: Class to decorate
        :param functions: Functions to possibly record
        :param resolution: ImplicitResolution providing a way to resolve the recording object to record to
        :return: Decorated class
        """

        for func_name, (func, bound_func) in variables(cls).items():
            if callable(bound_func) and func_name in functions:
                # Decorate functions to comply with recording
                setattr(cls, func_name,
                        protected_bindable(Recordable.wrap_method(func, functions[func_name], resolution)))

        # Auto-complete recording parameter as defined by the ImplicitResolution
        cls = Bindable.wrap_class(cls, ('recording', ))

        for func_name, (func, bound_func) in variables(cls).items():
            if callable(bound_func) and func_name in functions:
                # Remove recording from signature, so it will be invisible to the user
                class Wrapper(FunctionWrapper):
                    def __init__(self, function):
                        super(Wrapper, self).__init__(signature(argument_explicit_restrict('recording')(function)))
                        self.function = function

                    def __get__(self, instance, owner):
                        return argument_explicit_restrict('recording')(self.function.__get__(instance, owner))

                setattr(cls, func_name, Wrapper(func))

        return cls

    def __call__(self, cls):
        return Recordable.wrap_class(cls, self.functions, self.resolution)

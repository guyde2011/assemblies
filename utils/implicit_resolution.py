from functools import partial, wraps
from inspect import Parameter, Signature
from typing import Tuple, Any, Optional, Callable, TypeVar, Generic

from utils.class_manipulation import FunctionWrapper
from utils.argument_manipulation import signature

T = TypeVar('T')


class ImplicitResolution(Generic[T]):
    """
    Implicit argument resolution decorator.
    Allows arguments of functions to be resolved on run-time according to some implicit resolution function.
    """

    def __init__(self, resolve: Callable[[T, str], Tuple[bool, Optional[Any]]], *params: str):
        """
        Creates an implicit resolution decorator
        :param resolve: Mapping from (instance, param_name) -> (found, implicit_value)
        :param params: Parameters to allow implicit resolution for
        """
        self.params: Tuple[str, ...] = params
        self.resolve: Callable[[T, str], Tuple[bool, Optional[Any]]] = resolve

    @staticmethod
    def wrap_function(function, resolve: Callable[[T, str], Tuple[bool, Optional[Any]]],
                      param_names: Tuple[str, ...]):
        """
        Wraps a function to comply with the implicit resolution decoration, auto-fills resolved parameters
        :param function: Function to wrap
        :param resolve: Mapping from (instance, param_name) -> (found, implicit_value)
        :param param_names: Parameters to allow implicit resolution for
        :return: Wrapped function, support implicit resolution
        """
        # Get the signature of the function we are wrapping
        sig: Signature = signature(function)
        effective_params: Tuple[Parameter, ...] = tuple(param for name, param in sig.parameters.items()
                                                        if name != 'self')
        # Get parameters which can be resolved
        resolvable_params: Tuple[Parameter, ...] = tuple(param for param in effective_params
                                                         if param.name in param_names)
        resolvable_param_names: Tuple[str, ...] = tuple(param.name for param in resolvable_params)
        # Check for possibly resolvable parameters which are not keyword-only
        problem: Optional[Parameter] = next((param for param in resolvable_params
                                             if param.kind != Parameter.KEYWORD_ONLY), None)
        if problem:
            # Allow possibly resolvable parameters to be keyword-only, to avoid bugs and complications
            raise Exception(f"Cannot allow implicit resolution of parameter [{problem.name}] of [{function.__name__}]"
                            f", must be keyword-only")

        class Wrapper(FunctionWrapper):
            """
            Wrapper class to enable property functionality when all parameters are resolved
            """

            def __init__(self):
                super(Wrapper, self).__init__(sig)

            @staticmethod
            def get_partial_function(instance, owner, bound_method: bool):
                """
                Gives resolved parameters a default value, which is the parameter resolved with respect to the instance
                :param instance: Instance attempting to resolve function arguments for
                :param owner: Owner class
                :param bound_method: Flag indicating whether this should resemble a bound method
                :return: Partial function with resolved parameters already passed
                """
                _func = partial(function.__get__(instance if bound_method else None, owner),
                                **{name: value for name, (found, value) in
                                   {param_name: resolve(instance, param_name)
                                    for param_name in resolvable_param_names}.items()
                                   if found})

                # We want to preserve docstring and name, but new signature
                _sig = signature(_func, use_original=True)
                func = wraps(function)(_func)
                func.__signature__ = _sig

                return func

            def __get__(self, instance, owner):
                if instance is not None:
                    func = self.get_partial_function(instance, owner, False)
                    if all(param.default != Parameter.empty or name == 'self' for name, param
                           in signature(func).parameters.items()) and getattr(function, 'implicit_property', False):
                        # Check if function should be made a property when all parameters are resolved
                        # and check if all parameters are indeed resolved
                        return property(func).__get__(instance, owner)

                    return self.get_partial_function(instance, owner, True)

                return function.__get__(instance, owner)

        return Wrapper()

    def __call__(self, function):
        return self.wrap_function(function, self.resolve, self.params)


def implicit_property(function):
    """Decorator declaring a function to become a property once all parameters are resolved"""
    setattr(function, 'implicit_property', True)
    return function

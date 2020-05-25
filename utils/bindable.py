from typing import List, Optional, cast
from inspect import signature, Parameter, Signature, ismethod
from functools import partial


class Bindable:
    """
    Bindable decorator for classes
    This enables auto-filling parameters into the class functions
    For example, decorating a class Fooable with Bindable('bar'),
     which contains a function foo: (self, bar: int) -> 2 * bar, enables the following behaviour

    fooable = Fooable()
    fooable.bind(5)
    fooable.foo() == 10
    fooable.foo(bar=10) == 20
    """

    def __init__(self, *params: str):
        """
        Creates a bindable decorator
        :param params: Parameters available for instance binding
        """
        self.params: List[str] = list(params)

    @staticmethod
    def wrap_function(function, param_names: List[str]):
        """Wraps a function to comply with the bindable decoration, auto-fills bound parameters of the instance"""
        # Get the signature of the function we are wrapping
        sig: Signature = signature(function)
        effective_params = [param for name, param in sig.parameters.items() if name != 'self']
        # Get parameters which can be bound to the instance
        bindable_params: List[Parameter] = list(param for param in effective_params if param.name in param_names)
        bindable_param_names: List[str] = list(param.name for param in bindable_params)
        # Check for possibly bound parameters which are not keyword-only
        problem: Optional[Parameter] = next((param for param in bindable_params if param.kind != Parameter.KEYWORD_ONLY)
                                            , None)
        if problem:
            # Allow possibly bound parameters to be keyword-only, to avoid bugs and complications
            raise Exception(f"Cannot bind parameter [{problem.name}] of [{function.__name__}], must be keyword-only")

        class Wrapper:
            """
            Wrapper class to enable property functionality when all parameters are already bound
            """

            @staticmethod
            def get_partial_function(instance, owner, bound_method: bool):
                """Gives bound parameters a default value, which is the parameter bound to the instance"""
                return partial(function.__get__(instance if bound_method else None, owner),
                               **{name: value for name, value in getattr(instance, '_bound_params', {}).items()
                                  if name in bindable_param_names})

            def __get__(self, instance, owner):
                if instance is not None:
                    func = self.get_partial_function(instance, owner, False)
                    if all(param.default != Parameter.empty or name == 'self' for name, param
                           in signature(func).parameters.items()) and getattr(function, 'bindable_property', False):
                        # Check if function should be made a property when all parameters are bound
                        # and check if all parameters are bound
                        return property(func).__get__(instance, owner)

                    return self.get_partial_function(instance, owner, True)

                return function.__get__(instance, owner)

        return Wrapper()

    @staticmethod
    def wrap_class(cls, params: List[str]):
        """
        Wraps a class to comply with the bindable decoration: wraps all methods and adds bind & unbind functionality
        :param cls: Class to decorate
        :param params: Parameters to allow binding for
        :return: Decorated class
        """

        # Add an inner dictionary for storing bound parameters
        setattr(cls, '_bound_params', {})
        for name, func in vars(cls).items():
            # Decorate all non-protected functions
            if callable(func) and not name.startswith('_') and not isinstance(func, staticmethod):
                setattr(cls, name, Bindable.wrap_function(func, params))

        if len(params) == 1:
            # Only one possible bound parameter
            param: str = params[0]

            def bind(self, obj):
                """Binds {0} to this instance"""
                self._bound_params[param] = obj

            def unbind(self):
                """Unbinds {0} from the instance"""
                self._bound_params.pop(param)

            # Update docstrings to match bound parameter name
            bind.__doc__ = cast(bind.__doc__, str).format(param)
            unbind.__doc__ = cast(unbind.__doc__, str).format(param)
        else:
            # Many possible bound parameters
            def bind(self, **kwargs):  # type: ignore
                """Binds parameters to the instance"""
                problem: str = next((kwarg for kwarg in kwargs if kwarg not in params), None)
                if problem:
                    raise Exception(f"Cannot bind parameter [{problem}], was not declared bindable")

                for k, v in kwargs.items():
                    self._bound_params[k] = v

            def unbind(self, *names):  # type: ignore
                """Unbinds parameters from the instance"""
                problem: str = next((name for name in names if name not in params), None)
                if problem:
                    raise Exception(f"Cannot unbind parameter [{problem}], was not declared bindable")

                for name in names:
                    self._bound_params.pop(name)

        # Update binding signature to match possible bound parameters
        bind.__signature__ = signature(bind).replace(parameters=
                                                     [Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD)] +
                                                     [Parameter(name=name, kind=Parameter.KEYWORD_ONLY)
                                                      for name in params])

        # Add bind & unbind to the class
        setattr(cls, 'bind', bind)
        setattr(cls, 'unbind', unbind)

        return cls

    def __call__(self, cls):
        return self.wrap_class(cls, self.params)


def bindable_property(function):
    """Decorator declaring a function to become a property once all parameters are bound in the instance"""
    setattr(function, 'bindable_property', True)
    return function

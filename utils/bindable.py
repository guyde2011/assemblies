from collections.abc import Mapping
from functools import wraps, cached_property
from typing import Optional, Any, Tuple, Dict, Set, Generic, Iterator
from inspect import Parameter

from utils.class_manipulation import variables
from utils.argument_manipulation import signature
from utils.implicit_resolution import ImplicitResolution, implicit_property, T


class Bindable(Generic[T]):
    """
    Bindable decorator for classes
    This enables auto-filling parameters into the class functions
    For example, decorating a class Fooable with Bindable('bar'),
     which contains a function foo: (self, bar: int) -> 2 * bar, enables the following behaviour

    fooable = Fooable()
    fooable.bind(bar=5)
    fooable.foo() == 10
    fooable.foo(bar=10) == 20
    """

    def __init__(self, *params: str):
        """
        Creates a bindable decorator
        :param params: Parameters available for instance binding
        """
        self.params: Tuple[str, ...] = params

    @staticmethod
    def implicitly_resolve(instance: T, param_name: str) -> Tuple[bool, Optional[Any]]:
        """
        Attempt implicitly resolving a parameter, by checking if it was bound
        :param instance:
        :param param_name:
        :return:
        """
        _bound_params: Dict[str, Any] = getattr(instance, '_bound_params', {})
        return param_name in _bound_params, _bound_params.get(param_name, None)

    @staticmethod
    def implicitly_resolve_many(instances: Tuple[T], param_name: str, graceful: bool = True)\
            -> Tuple[bool, Optional[Any]]:
        """
        Attempt implicitly resolving many parameters, by checking if they all share a common bound value
        :param instances:
        :param param_name:
        :param graceful: Flag stating whether to throw an exception or to gracefully return nothing
        :return:
        """
        _options: Tuple[Tuple[bool, Optional[Any]], ...] = tuple(Bindable.implicitly_resolve(instance, param_name)
                                                                 for instance in instances)
        options: Set[Any] = set(implicit_value for found, implicit_value in _options if found)
        if len(options) == 1:
            return True, options.pop()
        elif len(options) > 1 and not graceful:
            raise Exception("Multiple different bound parameters for the different instances")

        return False, None

    @staticmethod
    def wrap_class(cls, params: Tuple[str, ...]):
        """
        Wraps a class to comply with the bindable decoration: wraps all methods and adds bind & unbind functionality
        :param cls: Class to decorate
        :param params: Parameters to allow binding for
        :return: Decorated class
        """
        implicit_resolution: ImplicitResolution = ImplicitResolution(Bindable.implicitly_resolve, *params)
        for func_name, (func, bound_func) in variables(cls).items():
            # Check function is not reserved
            if func_name in ('bind', 'unbind', 'bind_like', 'bound_params'):
                continue

            # Decorate all non-protected functions
            if callable(bound_func) and (not func_name.startswith('_') or getattr(func, 'override_protection', False))\
                    and not isinstance(bound_func, staticmethod):
                setattr(cls, func_name, implicit_resolution(func))

        # Update params to include previous bindings
        params: Tuple[str, ...] = tuple(set(getattr(cls, '_bindable_params', ()) + params))
        setattr(cls, '_bindable_params', params)

        original_init = getattr(cls, '__init__', lambda self, *args, **kwargs: None)

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Add _bound_params dictionary to instance
            original_init(self, *args, **kwargs)
            self._bound_params = {}

        setattr(cls, '__init__', new_init)

        # Many possible bound parameters
        def bind(self: T, **kwargs):
            """Binds (the) parameters {0} to the instance"""
            problem: str = next((kwarg for kwarg in kwargs if kwarg not in params), None)
            if problem:
                raise Exception(f"Cannot bind parameter [{problem}], was not declared bindable")

            for k, v in kwargs.items():
                self._bound_params[k] = v

        def bind_like(self: T, *others: T):
            """Binds this instance like another instance"""
            if len(others) == 0:
                self._bound_params = {}
                return

            self._bound_params = getattr(others[0], '_bound_params', {}).copy()
            for other in others[1:]:
                for key, value in getattr(other, 'bound_params', {}).items():
                    if key in self._bound_params and self._bound_params.get(key) != value:
                        self._bound_params.pop(key)

        def unbind(self: T, *names: str):
            """Unbinds (the) parameters {0} from the instance, pass no names to unbind all"""
            problem: str = next((name for name in names if name not in params), None)
            if problem:
                raise Exception(f"Cannot unbind parameter [{problem}], was not declared bindable")

            if len(names) == 0:
                names = tuple(self._bound_params.keys())

            for name in names:
                if name in self._bound_params:
                    self._bound_params.pop(name)

        @cached_property
        def bound_params(_self):
            class Proxy(Mapping):
                def __contains__(self, key: str):
                    return key in getattr(_self, '_bound_params', {})

                def __getitem__(self, key: str):
                    return getattr(_self, '_bound_params', {})[key]

                def __iter__(self):
                    return iter(getattr(_self, '_bound_params', {}))

                def __len__(self) -> int:
                    return len(list(iter(self)))

            return Proxy()

        bind.__doc__ = bind.__doc__.format(params)
        unbind.__doc__ = unbind.__doc__.format(params)

        # Update binding signature to match possible bound parameters
        bind.__signature__ = signature(bind).replace(parameters=
                                                     [Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD)] +
                                                     [Parameter(name=name, kind=Parameter.KEYWORD_ONLY)
                                                      for name in params])
        # Set name for cached property
        bound_params.__set_name__(cls, 'bound_params')

        # Add bind & unbind to the class
        setattr(cls, 'bind', bind)
        setattr(cls, 'unbind', unbind)
        setattr(cls, 'bind_like', bind_like)
        setattr(cls, 'bound_params', bound_params)

        return cls

    def __call__(self, cls):
        return self.wrap_class(cls, self.params)


def bindable_property(function):
    """Decorator declaring a function to become a property once all parameters are bound in the instance"""
    return implicit_property(function)


def protected_bindable(function):
    """Decorator declaring to attempt looking for bound parameters of a protected function"""
    setattr(function, 'override_protection', True)
    return function

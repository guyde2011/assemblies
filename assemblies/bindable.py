from typing import List, Optional, cast
from inspect import signature, Parameter, Signature
from functools import partial


class Bindable:
    def __init__(self, *params: str):
        self.params: List[str] = list(params)

    @staticmethod
    def wrap_function(function, param_names: List[str]):
        sig: Signature = signature(function)
        effective_params = [param for name, param in sig.parameters.items() if name != 'self']
        bindable_params: List[Parameter] = list(param for param in effective_params if param.name in param_names)
        bindable_param_names: List[str] = list(param.name for param in bindable_params)
        problem: Optional[Parameter] = next((param for param in bindable_params if param.kind != Parameter.KEYWORD_ONLY)
                                            , None)
        if problem:
            raise Exception(f"Cannot bind parameter [{problem.name}] of [{function.__name__}], must be keyword-only")

        class Wrapper:
            @staticmethod
            def get_partial_function(instance, owner, bound_method: bool):
                return partial(function.__get__(instance if bound_method else None, owner),
                               **{name: value for name, value in getattr(instance, '_bound_params', {}).items()
                                  if name in bindable_param_names})

            def __get__(self, instance, owner):
                if instance is not None:
                    func = self.get_partial_function(instance, owner, False)
                    if all(param.default != Parameter.empty or name == 'self' for name, param
                           in signature(func).parameters.items()) and getattr(function, 'bindable_property', False):
                        return property(func).__get__(instance, owner)

                    return self.get_partial_function(instance, owner, True)

                return function.__get__(instance, owner)

        return Wrapper()

    @staticmethod
    def wrap_class(cls, params: List[str]):
        setattr(cls, '_bound_params', {})
        for name, func in vars(cls).items():
            if callable(func) and not name.startswith('_'):
                setattr(cls, name, Bindable.wrap_function(func, params))

        if len(params) == 1:
            param: str = params[0]

            def bind(self, obj):
                """Binds {0} to this instance"""
                self._bound_params[param] = obj

            def unbind(self):
                """Unbinds {0} from the instance"""
                self._bound_params.pop(param)

            bind.__doc__ = cast(bind.__doc__, str).format(param)
            unbind.__doc__ = cast(unbind.__doc__, str).format(param)
        else:
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

        bind.__signature__ = signature(bind).replace(parameters=
                                                     [Parameter(name='self', kind=Parameter.POSITIONAL_OR_KEYWORD)] +
                                                     [Parameter(name=name, kind=Parameter.KEYWORD_ONLY)
                                                      for name in params])
        setattr(cls, 'bind', bind)
        setattr(cls, 'unbind', unbind)

        return cls

    def __call__(self, cls):
        return self.wrap_class(cls, self.params)


def bindable_property(function):
    setattr(function, 'bindable_property', True)
    return function

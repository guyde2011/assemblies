from inspect import Signature
from typing import Any, Dict, Tuple


def variables(cls):
    result: Dict[str, Tuple[Any, Any]] = {}
    for name, attr in vars(cls).items():
        bound = getattr(cls, name)
        if isinstance(attr, staticmethod):
            result[name] = bound, bound
        else:
            result[name] = attr, bound

    return result


class FunctionWrapper:
    def __init__(self, signature: Signature):
        self.__signature__ = signature

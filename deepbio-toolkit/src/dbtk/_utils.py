import functools
import importlib
import inspect
import sys
from typing import Callable, ParamSpec, Type, TypeVar
import warnings

# Decorators ---------------------------------------------------------------------------------------

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


string_types = (type(b''), type(u''))


# https://stackoverflow.com/a/40301488
def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    def decorator(f):
        return f
    return decorator

    # if isinstance(reason, string_types):

    #     # The @deprecated is used with a 'reason'.
    #     #
    #     # .. code-block:: python
    #     #
    #     #    @deprecated("please, use another function")
    #     #    def old_function(x, y):
    #     #      pass

    #     def decorator(func1):

    #         if inspect.isclass(func1):
    #             fmt1 = "Call to deprecated class {name} ({reason})."
    #         else:
    #             fmt1 = "Call to deprecated function {name} ({reason})."

    #         @functools.wraps(func1)
    #         def new_func1(*args, **kwargs):
    #             warnings.simplefilter('always', DeprecationWarning)
    #             warnings.warn(
    #                 fmt1.format(name=func1.__name__, reason=reason),
    #                 category=DeprecationWarning,
    #                 stacklevel=2
    #             )
    #             warnings.simplefilter('default', DeprecationWarning)
    #             return func1(*args, **kwargs)

    #         return new_func1

    #     return decorator

    # elif inspect.isclass(reason) or inspect.isfunction(reason):

    #     # The @deprecated is used without any 'reason'.
    #     #
    #     # .. code-block:: python
    #     #
    #     #    @deprecated
    #     #    def old_function(x, y):
    #     #      pass

    #     func2 = reason

    #     if inspect.isclass(func2):
    #         fmt2 = "Call to deprecated class {name}."
    #     else:
    #         fmt2 = "Call to deprecated function {name}."

    #     @functools.wraps(func2)
    #     def new_func2(*args, **kwargs):
    #         warnings.simplefilter('always', DeprecationWarning)
    #         warnings.warn(
    #             fmt2.format(name=func2.__name__),
    #             category=DeprecationWarning,
    #             stacklevel=2
    #         )
    #         warnings.simplefilter('default', DeprecationWarning)
    #         return func2(*args, **kwargs)

    #     return new_func2

    # else:
    #     raise TypeError(repr(type(reason)))


def export(fn):
    """Use a decorator to avoid retyping function/class names.

    * Based on an idea by Duncan Booth:
      http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    * Improved via a suggestion by Dave Angel:
      http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        name = fn.__name__
        all_ = mod.__all__
        if name not in all_:
            all_.append(name)
    else:
        mod.__all__ = [fn.__name__] # type: ignore
    return fn

def load_class(class_path: str) -> Type:
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

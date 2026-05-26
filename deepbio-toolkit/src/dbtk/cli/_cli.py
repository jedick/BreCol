import abc
import argparse
import sys
from typing import Optional

def subcommand(subcommand_name: str, help: Optional[str] = None):
    def decorator(fn):
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
        return fn(subcommand_name, help)
    return decorator

class CliSubCommand(abc.ABC):
    def __init__(self, name, doc):
        super().__init__()
        self.name = name
        self.__doc__ = doc

    def configure(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abc.abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        raise NotImplementedError

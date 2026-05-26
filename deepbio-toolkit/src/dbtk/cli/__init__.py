import argparse
import importlib
import pkgutil
import sys

def main():
    from . import subcommands
    module_names = [name for _, name, _ in pkgutil.iter_modules(subcommands.__path__)]
    modules = {}
    parser = argparse.ArgumentParser(description="DeepBio Toolkit CLI")
    subparsers = parser.add_subparsers(dest="module", title="module", required=True)
    for module_name in module_names:
        modules[module_name] = {
            "module": importlib.import_module(f".{module_name}", f"dbtk.cli.subcommands"),
            "subcommands": {}
        }
        module = modules[module_name]["module"]
        module_parser = subparsers.add_parser(module_name, help=module.__doc__)
        module_subparsers = module_parser.add_subparsers(dest="subcommand", title="subcommand", required=True)
        for subcommand_cls_name in module.__all__:
            instance = getattr(module, subcommand_cls_name)
            instance.configure(module_subparsers.add_parser(instance.name, help=instance.__doc__))
            modules[module_name]["subcommands"][instance.name] = instance
    args = parser.parse_args()
    sys.exit(modules[args.module]["subcommands"][args.subcommand].run(args) or 0)

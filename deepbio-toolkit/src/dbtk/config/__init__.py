import importlib
import json
import os
from pathlib import Path
import re
from transformers import PreTrainedModel
from typing import Any, Dict, Set, List, Optional, Type, Union
import yaml


def import_class(class_path: str) -> type:
    """Import a class from its fully qualified path.

    Args:
        class_path: Fully qualified path to the class (e.g. 'package.module.ClassName')

    Returns:
        The class type object

    Raises:
        ImportError: If the module or class cannot be imported
        AttributeError: If the class doesn't exist in the module
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def is_object_definition(value: Any) -> bool:
    """Check if a value represents an object definition.

    An object definition must be a dictionary containing both 'class_path' and
    'config' keys.

    Args:
        value: The value to check

    Returns:
        bool: True if the value is an object definition, False otherwise
    """
    # First check if it's a dictionary and has at least one of the required keys
    if not isinstance(value, dict) or ('class_path' not in value and 'config' not in value):
        return False

    # If it has one of the keys, it's trying to be an object definition
    # so make sure it has both required keys
    if 'class_path' not in value:
        raise TypeError("Object definition must contain 'class_path'")
    if 'config' not in value:
        raise TypeError("Object definition must contain 'config'")

    return True


class ConfigNode:
    """Represents a node in the configuration dependency graph."""
    def __init__(self, path: str, config: Any):
        self.path = path
        self.config = config
        self.dependencies: Set[str] = set()
        self._find_dependencies(config)
        self.instance: Optional[Any] = None

    def _find_dependencies(self, value: Any) -> None:
        """Recursively find all dependencies in the configuration."""
        if isinstance(value, str):
            # Look for ${path.to.value} and ${env:VAR_NAME} patterns
            matches = re.finditer(r'\${([^}]+)}', value)
            for m in matches:
                ref = m.group(1)
                if not ref.startswith('env:'):  # Only add non-env refs as dependencies
                    self.dependencies.update(ref.split('.')[0:1])
        elif isinstance(value, dict):
            for v in value.values():
                self._find_dependencies(v)
        elif isinstance(value, list):
            for v in value:
                self._find_dependencies(v)

class ConfigurationManager:
    """Manages configuration parsing and instantiation with dependencies."""
    def __init__(self):
        self.nodes: Dict[str, ConfigNode] = {}
        self.instances: Dict[str, Any] = {}

    def _is_pretrained_model(self, cls: Type) -> bool:
        """Check if a class inherits from PreTrainedModel."""
        try:
            return issubclass(cls, PreTrainedModel)
        except TypeError:
            return False

    def _resolve_references(self, value: Any) -> Any:
        """Resolve ${path.to.value} references in the configuration.

        This handles both string interpolation (when the reference is part of a larger string)
        and direct object references (when the string is exactly a reference).

        Examples:
            ${model.tokenizer} -> returns the actual tokenizer object
            "vocab ${model.vocab_size}" -> returns the string with vocab_size converted to str
        """
        if isinstance(value, str):
            # If the entire string is a reference, return the actual object
            if value.startswith('${') and value.endswith('}') and value.count('${') == 1:
                ref = value[2:-1]
                if ref.startswith('env:'):
                    env_var = ref[4:]  # Remove 'env:' prefix
                    if env_var not in os.environ:
                        raise KeyError(f"Environment variable '{env_var}' not found")
                    return os.environ[env_var]
                else:
                    path = ref.split('.')
                    obj = self.instances[path[0]]
                    for attr in path[1:]:
                        obj = getattr(obj, attr)
                    return obj

            # Otherwise, do string interpolation
            def replace_ref(match):
                ref = match.group(1)
                if ref.startswith('env:'):
                    env_var = ref[4:]  # Remove 'env:' prefix
                    if env_var not in os.environ:
                        raise KeyError(f"Environment variable '{env_var}' not found")
                    return os.environ[env_var]
                else:
                    path = ref.split('.')
                    obj = self.instances[path[0]]
                    for attr in path[1:]:
                        obj = getattr(obj, attr)
                    return str(obj)

            # Replace all references in the string
            while '${' in value:
                value = re.sub(r'\${([^}]+)}', replace_ref, value)
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_references(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_references(v) for v in value]
        return value

    def _build_dependency_graph(self, config: Dict[str, Any]) -> None:
        """Build the dependency graph from the configuration."""
        # Create nodes for each top-level config
        for key, value in config.items():
            self.nodes[key] = ConfigNode(key, value)

    def _get_instantiation_order(self) -> List[str]:
        """Determine the order in which to instantiate objects using topological sort."""
        visited = set()
        temp_mark = set()
        order = []

        def visit(node_id: str) -> None:
            if node_id in temp_mark:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id not in visited:
                temp_mark.add(node_id)
                node = self.nodes[node_id]
                for dep in node.dependencies:
                    if dep in self.nodes:  # Only visit if it's a config node
                        visit(dep)
                temp_mark.remove(node_id)
                visited.add(node_id)
                order.append(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        return order

    def _parse_config(self, config: Any) -> Any:
        """Parse configuration dictionary, handling nested object definitions."""
        if isinstance(config, (str, int, float, bool, type(None))):
            return config
        elif isinstance(config, list):
            return [self._parse_config(item) for item in config]
        elif not isinstance(config, dict):
            return config

        result = {}
        for key, value in config.items():
            if isinstance(value, dict):
                if is_object_definition(value):
                    if 'class_path' not in value:
                        raise TypeError("Object definition must contain 'class_path'")
                    if 'config' not in value:
                        raise TypeError("Object definition must contain 'config'")

                    # Recursively parse nested object definition
                    parsed_config = self._parse_config(value['config'])
                    resolved_config = self._resolve_references(parsed_config)
                    cls = import_class(value['class_path'])

                    if self._is_pretrained_model(cls):
                        config_cls = cls.config_class
                        if config_cls is None:
                            raise ValueError(
                                f"Model {cls.__name__} is a PreTrainedModel but has no config_class"
                            )
                        result[key] = cls(config_cls(**resolved_config))
                    else:
                        result[key] = cls(**resolved_config)
                else:
                    # Recursively parse nested dictionary
                    result[key] = self._parse_config(value)
            elif isinstance(value, list):
                # Recursively parse list items
                result[key] = [self._parse_config(item) for item in value]
            else:
                result[key] = value
        return result

    def parse(self, config: Any) -> Any:
        """Parse and instantiate a configuration with dependencies.

        Args:
            config: Configuration to parse. Can be a dictionary containing object
                definitions, a list of configurations, or a simple value.

        Returns:
            Parsed configuration with all references resolved and objects instantiated
        """
        # Handle simple values
        if isinstance(config, (str, int, float, bool, type(None))):
            return self._resolve_references(config)
        elif isinstance(config, list):
            return [self.parse(item) for item in config]

        # Handle dictionaries
        if not isinstance(config, dict):
            return config

        # Handle object definition
        if is_object_definition(config):
            # Parse the config section
            parsed_config = self._parse_config(config['config'])
            resolved_config = self._resolve_references(parsed_config)

            # Import and instantiate the class
            cls = import_class(config['class_path'])
            if self._is_pretrained_model(cls):
                # Get the config class for this model
                config_cls = cls.config_class
                if config_cls is None:
                    raise ValueError(
                        f"Model {cls.__name__} is a PreTrainedModel but has no config_class"
                    )

                # Create the config instance
                config = config_cls(**resolved_config)

                # Instantiate the model with its config
                return cls(config)
            else:
                # Normal instantiation for other classes
                return cls(**resolved_config)

        # Build dependency graph
        self._build_dependency_graph(config)

        # Get instantiation order
        order = self._get_instantiation_order()

        # Process nodes in order
        for node_id in order:
            node = self.nodes[node_id]
            if is_object_definition(node.config):
                # Parse the config section
                parsed_config = self._parse_config(node.config['config'])
                resolved_config = self._resolve_references(parsed_config)

                # Import and instantiate the class
                cls = import_class(node.config['class_path'])
                if self._is_pretrained_model(cls):
                    # Get the config class for this model
                    config_cls = cls.config_class
                    if config_cls is None:
                        raise ValueError(
                            f"Model {cls.__name__} is a PreTrainedModel but has no config_class"
                        )

                    # Create the config instance
                    config = config_cls(**resolved_config)

                    # Instantiate the model with its config
                    self.instances[node_id] = cls(config)
                else:
                    # Normal instantiation for other classes
                    self.instances[node_id] = cls(**resolved_config)
            else:
                # For non-object definitions, parse nested objects and resolve references
                parsed_config = self._parse_config(node.config)
                self.instances[node_id] = self._resolve_references(parsed_config)

        return self.instances


def parse(config: Any) -> Any:
    """Parse a configuration with dependency resolution.

    Args:
        config: The configuration to parse. Can be:
            - A dictionary of named configurations
            - An object definition dictionary with 'class_path' and 'config'
            - A list of configurations
            - A simple value with possible ${references}

    Returns:
        The parsed and instantiated configuration

    Raises:
        ValueError: If circular dependencies are detected
        ImportError: If a class cannot be imported
        TypeError: If configuration is invalid
    """
    return ConfigurationManager().parse(config)


def merge(config: dict, other: dict):
    """
    Recursively merge dictionaries
    """
    for key, value in other.items():
        if isinstance(value, dict):
            if key not in config:
                config[key] = {}
            merge(config[key], value)
        else:
            config[key] = value


def load(*configs: Union[str, Path]):
    merged_config = {}
    for config in configs:
        if isinstance(config, str):
            config = Path(config)
        if not config.exists():
            raise FileNotFoundError(f"Config file {config} does not exist")
        with open(config, 'r') as f:
            if config.suffix == ".json":
                config = json.load(f)
            elif config.suffix in (".yaml", ".yml"):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config.suffix}")
            merge(merged_config, config)
    return parse(merged_config)

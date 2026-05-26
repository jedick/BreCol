import os
import unittest
from dbtk.config import parse, ConfigurationManager

class SimpleClass:
    def __init__(self, value):
        self.value = value

    @property
    def doubled(self):
        return self.value * 2

class ComplexClass:
    def __init__(self, simple, name):
        self.simple = simple
        self.name = name

class TestConfig(unittest.TestCase):
    def test_simple_value(self):
        """Test parsing of simple values without references."""
        self.assertEqual(parse(42), 42)
        self.assertEqual(parse("hello"), "hello")
        self.assertEqual(parse([1, 2, 3]), [1, 2, 3])
        self.assertEqual(parse({"a": 1, "b": 2}), {"a": 1, "b": 2})

    def test_object_definition(self):
        """Test parsing of object definitions."""
        config = {
            "class_path": "test.test_config.SimpleClass",
            "config": {"value": 42}
        }
        obj = parse(config)
        self.assertIsInstance(obj, SimpleClass)
        self.assertEqual(obj.value, 42)

    def test_nested_object_definition(self):
        """Test parsing of nested object definitions."""
        config = {
            "class_path": "test.test_config.ComplexClass",
            "config": {
                "simple": {
                    "class_path": "test.test_config.SimpleClass",
                    "config": {"value": 42}
                },
                "name": "test"
            }
        }
        obj = parse(config)
        self.assertIsInstance(obj, ComplexClass)
        self.assertIsInstance(obj.simple, SimpleClass)
        self.assertEqual(obj.simple.value, 42)
        self.assertEqual(obj.name, "test")

    def test_references(self):
        """Test reference resolution in configs."""
        config = {
            "simple": {
                "class_path": "test.test_config.SimpleClass",
                "config": {"value": 42}
            },
            "complex": {
                "class_path": "test.test_config.ComplexClass",
                "config": {
                    "simple": "${simple}",
                    "name": "Value is ${simple.value}, doubled is ${simple.doubled}"
                }
            }
        }
        result = parse(config)
        self.assertIsInstance(result["complex"], ComplexClass)
        self.assertIs(result["complex"].simple, result["simple"])  # Reference preserved
        self.assertEqual(result["complex"].name, "Value is 42, doubled is 84")

    def test_env_vars(self):
        """Test environment variable substitution."""
        os.environ["TEST_VALUE"] = "42"
        os.environ["TEST_PREFIX"] = "test"

        # Direct reference
        config = {"value": "${env:TEST_VALUE}"}
        self.assertEqual(parse(config), {"value": "42"})

        # String interpolation
        config = {"path": "/path/${env:TEST_PREFIX}/data"}
        self.assertEqual(parse(config), {"path": "/path/test/data"})

        # Nested in object definition
        config = {
            "class_path": "test.test_config.SimpleClass",
            "config": {"value": "${env:TEST_VALUE}"}
        }
        obj = parse(config)
        self.assertEqual(obj.value, "42")  # Note: value is string since env vars are strings

    def test_missing_env_var(self):
        """Test error on missing environment variable."""
        config = {"value": "${env:NONEXISTENT_VAR}"}
        with self.assertRaisesRegex(KeyError, "Environment variable 'NONEXISTENT_VAR' not found"):
            parse(config)

    def test_circular_dependencies(self):
        """Test error on circular dependencies."""
        config = {
            "a": {
                "class_path": "test.test_config.SimpleClass",
                "config": {"value": "${b.value}"}
            },
            "b": {
                "class_path": "test.test_config.SimpleClass",
                "config": {"value": "${a.value}"}
            }
        }
        with self.assertRaisesRegex(ValueError, "Circular dependency detected"):
            parse(config)

    def test_invalid_reference(self):
        """Test error on invalid reference."""
        config = {
            "simple": {
                "class_path": "test.test_config.SimpleClass",
                "config": {"value": 42}
            },
            "ref": "${simple.nonexistent}"
        }
        with self.assertRaises(AttributeError):
            parse(config)

    def test_invalid_class_path(self):
        """Test error on invalid class path."""
        config = {
            "class_path": "nonexistent.module.Class",
            "config": {}
        }
        with self.assertRaises(ImportError):
            parse(config)

    def test_missing_config(self):
        """Test error on missing config in object definition."""
        config = {
            "class_path": "test.test_config.SimpleClass"
        }
        with self.assertRaisesRegex(TypeError, "Object definition must contain 'config'"):
            parse(config)

    def test_missing_class_path(self):
        """Test error on missing class_path in object definition."""
        config = {
            "config": {"value": 42}
        }
        with self.assertRaisesRegex(TypeError, "Object definition must contain 'class_path'"):
            parse(config)

if __name__ == '__main__':
    unittest.main()

"""A small, strict JSON-Schema subset for nested tool arguments.

Schemas are compiled when a ToolSpec is built, so a typo or unsupported keyword
fails at registration instead of silently disabling a check. Two choices are
stricter than JSON Schema: objects are closed unless ``additionalProperties``
says otherwise, and booleans never count as integers or numbers.
"""

import re
from typing import Any, Callable

TYPES = {
    "string": lambda v: type(v) is str,
    "integer": lambda v: type(v) is int,
    "number": lambda v: type(v) in (int, float),
    "boolean": lambda v: type(v) is bool,
    "array": lambda v: type(v) is list,
    "object": lambda v: type(v) is dict,
    "null": lambda v: v is None,
}
KEYWORDS = {
    "type", "enum", "const",
    "minLength", "maxLength", "pattern",
    "minimum", "maximum",
    "items", "minItems", "maxItems", "uniqueItems",
    "properties", "required", "additionalProperties",
}
COUNTS = ("minLength", "maxLength", "minItems", "maxItems")


class SchemaError(ValueError):
    """An argument does not match its schema; the message names only the path."""


def compile_schema(schema: dict) -> Callable[[Any], None]:
    """Validate ``schema`` now and return a checker that raises SchemaError."""
    return _compile(schema, "$")


def _compile(schema, path):
    if type(schema) is not dict:
        raise ValueError(f"Schema at {path} must be an object")
    unknown = set(schema) - KEYWORDS
    if unknown:
        raise ValueError(f"Unsupported schema keywords at {path}: {sorted(unknown)}")
    checks = []

    if "type" in schema:
        kinds = schema["type"] if type(schema["type"]) is list else [schema["type"]]
        if not kinds or any(kind not in TYPES for kind in kinds):
            raise ValueError(f"Unknown type at {path}")
        tests = [TYPES[kind] for kind in kinds]
        checks.append((lambda v: any(test(v) for test in tests), "type"))
    if "enum" in schema:
        if type(schema["enum"]) is not list or not schema["enum"]:
            raise ValueError(f"enum at {path} must be a non-empty list")
        allowed = list(schema["enum"])
        checks.append((lambda v: any(_same(v, a) for a in allowed), "enum"))
    if "const" in schema:
        constant = schema["const"]
        checks.append((lambda v: _same(v, constant), "const"))
    for key in COUNTS:
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"{key} at {path} must be a nonnegative integer")
    for key in ("minimum", "maximum"):
        if key in schema and type(schema[key]) not in (int, float):
            raise ValueError(f"{key} at {path} must be a number")

    def bound(key, measure, compare, applies):
        if key in schema:
            limit = schema[key]
            checks.append((lambda v: not applies(v) or compare(measure(v), limit), key))

    is_str, is_list = TYPES["string"], TYPES["array"]
    is_num = TYPES["number"]
    bound("minLength", len, lambda a, b: a >= b, is_str)
    bound("maxLength", len, lambda a, b: a <= b, is_str)
    bound("minItems", len, lambda a, b: a >= b, is_list)
    bound("maxItems", len, lambda a, b: a <= b, is_list)
    bound("minimum", lambda v: v, lambda a, b: a >= b, is_num)
    bound("maximum", lambda v: v, lambda a, b: a <= b, is_num)
    if "pattern" in schema:
        pattern = re.compile(schema["pattern"])
        checks.append((lambda v: not is_str(v) or pattern.fullmatch(v) is not None, "pattern"))
    if schema.get("uniqueItems") is True:
        checks.append((lambda v: not is_list(v) or _unique(v), "uniqueItems"))

    items = _compile(schema["items"], f"{path}[]") if "items" in schema else None
    properties = {
        name: _compile(sub, f"{path}.{name}")
        for name, sub in schema.get("properties", {}).items()
    }
    required = schema.get("required", [])
    if type(required) is not list or any(type(name) is not str for name in required):
        raise ValueError(f"required at {path} must be a list of names")
    extra = schema.get("additionalProperties", False)
    extra = _compile(extra, f"{path}.*") if type(extra) is dict else extra
    if type(extra) is not bool and not callable(extra):
        raise ValueError(f"additionalProperties at {path} must be a boolean or schema")

    def check(value, where=path):
        for test, keyword in checks:
            if not test(value):
                raise SchemaError(f"{where} fails {keyword}")
        if items is not None and is_list(value):
            for index, item in enumerate(value):
                items(item, f"{where}[{index}]")
        if TYPES["object"](value):
            missing = [name for name in required if name not in value]
            if missing:
                raise SchemaError(f"{where} is missing required properties")
            for name, item in value.items():
                if name in properties:
                    properties[name](item, f"{where}.{name}")
                elif extra is False:
                    raise SchemaError(f"{where} has an unexpected property")
                elif callable(extra):
                    extra(item, f"{where}.*")

    return check


def _same(a, b):
    """JSON equality that keeps booleans apart from numbers, at any depth."""
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    if type(a) is list and type(b) is list:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a) is dict and type(b) is dict:
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if type(a) in (list, dict) or type(b) in (list, dict):
        return False
    return a == b


def _unique(values):
    seen = []
    for value in values:
        if any(_same(value, other) for other in seen):
            return False
        seen.append(value)
    return True

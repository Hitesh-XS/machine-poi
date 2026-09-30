"""Nested argument validation: the JSON-Schema subset and ToolSpec validators."""

import asyncio

import pytest

from machine_poi.guardian import (
    ActionScope,
    Gateway,
    ProposedAction,
    RunState,
    TaskGrant,
    ToolSpec,
    Verdict,
)
from machine_poi.guardian.schema import SchemaError, compile_schema

EMAIL = {
    "type": "object",
    "properties": {
        "to": {
            "type": "array",
            "items": {"type": "string", "pattern": r"[a-z]+@example\.org"},
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
        },
        "priority": {"type": "integer", "minimum": 1, "maximum": 3},
        "attachments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string", "maxLength": 20}},
                "required": ["name"],
            },
        },
    },
    "required": ["to"],
}


def test_valid_nested_values_pass():
    check = compile_schema(EMAIL)
    check({"to": ["ana@example.org"], "priority": 2, "attachments": [{"name": "a.pdf"}]})


@pytest.mark.parametrize("value, where", [
    ({"to": ["mallory@evil.com"]}, "$.to[0] fails pattern"),
    ({"to": []}, "$.to fails minItems"),
    ({"to": ["a@example.org", "a@example.org"]}, "$.to fails uniqueItems"),
    ({"to": ["a@example.org"], "priority": True}, "$.priority fails type"),
    ({"to": ["a@example.org"], "priority": 9}, "$.priority fails maximum"),
    ({"to": ["a@example.org"], "bcc": ["x@example.org"]}, "unexpected property"),
    ({"to": ["a@example.org"], "attachments": [{}]}, "missing required"),
    ({"to": ["a@example.org"], "attachments": [{"name": "n", "path": "/etc"}]},
     "unexpected property"),
    ({"to": ["a@example.org"], "attachments": [{"name": "x" * 21}]}, "maxLength"),
    ({"priority": 1}, "missing required"),
])
def test_rejected_nested_fields_name_the_path_not_the_value(value, where):
    with pytest.raises(SchemaError, match=where.replace("[", r"\[").replace("$", r"\$")) as err:
        compile_schema(EMAIL)(value)
    assert "evil" not in str(err.value) and "/etc" not in str(err.value)


def test_enum_const_and_open_objects():
    assert compile_schema({"enum": [[1, 2], "x"]})([1, 2]) is None
    with pytest.raises(SchemaError):
        compile_schema({"enum": [[1, 2]]})([True, 2])  # booleans are not numbers
    with pytest.raises(SchemaError):
        compile_schema({"const": 0})(False)
    open_object = compile_schema({"type": "object", "additionalProperties": {"type": "string"}})
    open_object({"any": "text"})
    with pytest.raises(SchemaError):
        open_object({"any": 1})
    compile_schema({"type": "object", "additionalProperties": True})({"any": [1]})


@pytest.mark.parametrize("schema", [
    {"type": "strng"},
    {"maxLenght": 3},  # a typo must not disable the check
    {"enum": []},
    {"minItems": -1},
    {"properties": {"a": {"format": "email"}}},
    {"required": "a"},
    "not a schema",
])
def test_bad_schemas_fail_at_registration(schema):
    with pytest.raises(ValueError):
        compile_schema(schema)


def tool(validators):
    async def execute(args, context):
        effects.append(args)
        return {}

    return ToolSpec(
        "email",
        "1",
        {"message": dict, "note": str},
        lambda args: ActionScope(frozenset({"mailbox"}), frozenset({"smtp"})),
        execute,
        validators=validators,
    )


effects = []


def test_tool_spec_rejects_invalid_nested_arguments_before_any_effect():
    async def scenario():
        effects.clear()
        gateway = Gateway(
            [tool({"message": EMAIL, "note": lambda note: len(note) < 50})],
            operators={"operator"},
            clock=lambda: 0.0,
        )
        gateway.issue("operator", TaskGrant(
            "run", "agent", 100, frozenset({"email"}), frozenset({"mailbox"}),
            frozenset({"smtp"}),
        ))

        def action(name, message, note="hi"):
            return ProposedAction.create("run", name, "email", {"message": message, "note": note})

        ok = await gateway.submit("agent", action("a", {"to": ["ana@example.org"]}))
        assert ok.status == "executed" and len(effects) == 1
        assert gateway.preview("agent", action("b", {"to": ["ana@example.org"]}, "x" * 60)).reason == (
            "invalid_arguments"
        )
        bad = await gateway.submit("agent", action("c", {"to": ["mallory@evil.com"]}))
        assert bad.decision.verdict == Verdict.DENY
        assert bad.decision.reason == "invalid_arguments" and len(effects) == 1
        assert gateway.state("run") == RunState.STOPPED

    asyncio.run(scenario())


def test_validators_must_name_known_arguments_and_be_usable():
    with pytest.raises(ValueError, match="unknown arguments"):
        tool({"attachments": {"type": "array"}})
    with pytest.raises(ValueError, match="schema dict or a callable"):
        tool({"note": "short"})
    with pytest.raises(ValueError, match="Unsupported schema keywords"):
        tool({"message": {"typ": "object"}})

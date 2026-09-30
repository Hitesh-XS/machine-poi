"""Deterministic policy, evaluated on trusted adapter metadata."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Optional

from .contracts import (
    ActionScope,
    Decision,
    ProposedAction,
    TaskGrant,
    Verdict,
    digest,
    identifier,
)
from .schema import compile_schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    argument_types: dict[str, type]
    describe: Callable[[dict], ActionScope]
    execute: Callable[[dict, object], Awaitable[object]]
    # Optional nested checks per argument: a JSON-Schema subset (see
    # guardian/schema.py) or a callable that raises, or returns False, to reject.
    validators: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        identifier(self.name)
        identifier(self.version)
        allowed = {str, int, float, bool, list, dict}
        if any(kind not in allowed for kind in self.argument_types.values()):
            raise ValueError("Unsupported schema type")
        object.__setattr__(
            self, "argument_types", MappingProxyType(dict(self.argument_types))
        )
        validators = dict(self.validators or {})
        if not set(validators) <= set(self.argument_types):
            raise ValueError("Validators name unknown arguments")
        checks = {
            name: compile_schema(rule) if isinstance(rule, dict) else rule
            for name, rule in validators.items()
        }
        if any(not callable(check) for check in checks.values()):
            raise ValueError("A validator must be a schema dict or a callable")
        object.__setattr__(self, "validators", MappingProxyType(checks))

    def resolve(self, action: ProposedAction) -> ActionScope:
        args = action.arguments
        if set(args) != set(self.argument_types):
            raise ValueError("Missing or unknown tool argument")
        if any(
            type(args[key]) is not kind for key, kind in self.argument_types.items()
        ):
            raise ValueError("Wrong tool argument type")
        for name, check in self.validators.items():
            if check(args[name]) is False:
                raise ValueError("Tool argument failed validation")
        scope = self.describe(args)
        if type(scope) is not ActionScope:
            raise ValueError("Adapter must resolve an ActionScope")
        return scope


def binding(action, scope, tool, grant):
    return digest(
        {
            "action": action.fingerprint,
            "scope": scope.record(),
            "tool_version": tool.version,
            "policy_version": grant.policy_version,
        }
    )


class Policy:
    version = "1"

    def evaluate(
        self,
        grant: TaskGrant,
        action: ProposedAction,
        scope: ActionScope,
        tool: ToolSpec,
        now: float,
    ) -> Decision:
        fingerprint = binding(action, scope, tool, grant)
        reason = None
        if grant.policy_version != self.version:
            reason = "policy_version"
        elif now >= grant.expires_at:
            reason = "grant_expired"
        elif action.tool not in grant.tools:
            reason = "tool_scope"
        elif not scope.resources <= grant.resources:
            reason = "resource_scope"
        elif not scope.destinations <= grant.destinations:
            reason = "destination_scope"
        elif not scope.data_classes <= grant.data_classes:
            reason = "data_scope"
        if reason:
            return Decision(Verdict.DENY, reason, fingerprint, self.version)
        return Decision(
            Verdict.REVIEW if scope.requires_review else Verdict.ALLOW,
            "approval_required" if scope.requires_review else "within_grant",
            fingerprint,
            self.version,
        )

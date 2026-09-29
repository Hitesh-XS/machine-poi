"""Deterministic policy, evaluated on trusted adapter metadata."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Awaitable, Callable

from .contracts import (
    ActionScope,
    Decision,
    ProposedAction,
    TaskGrant,
    Verdict,
    digest,
    identifier,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    argument_types: dict[str, type]
    describe: Callable[[dict], ActionScope]
    execute: Callable[[dict, object], Awaitable[object]]

    def __post_init__(self):
        identifier(self.name)
        identifier(self.version)
        allowed = {str, int, float, bool, list, dict}
        if any(kind not in allowed for kind in self.argument_types.values()):
            raise ValueError("Unsupported schema type")
        object.__setattr__(
            self, "argument_types", MappingProxyType(dict(self.argument_types))
        )

    def resolve(self, action: ProposedAction) -> ActionScope:
        args = action.arguments
        if set(args) != set(self.argument_types):
            raise ValueError("Missing or unknown tool argument")
        if any(
            type(args[key]) is not kind for key, kind in self.argument_types.items()
        ):
            raise ValueError("Wrong tool argument type")
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

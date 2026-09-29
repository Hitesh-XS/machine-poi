"""Immutable host grants and JSON action proposals. No model dependencies."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise ValueError("Invalid identifier")


def nonnegative(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("Budget must be a nonnegative integer")


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"
    STOP = "stop"


class RunState(str, Enum):
    RUNNING = "running"
    OBSERVE = "observe"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True)
class TaskGrant:
    """Issued in the trusted host, never deserialized from an agent request.

    Resources and destinations are exact identifiers resolved by a trusted tool
    adapter. An empty set permits none, never everything. Costs use host-defined
    integer units and reserve a conservative upper bound before dispatch.
    """

    run_id: str
    principal: str
    expires_at: float
    tools: frozenset[str]
    resources: frozenset[str]
    destinations: frozenset[str] = frozenset()
    data_classes: frozenset[str] = frozenset({"public"})
    max_actions: int = 10
    max_attempts: int = 30
    max_cost_units: int = 0
    max_tokens: int = 0
    policy_version: str = "1"
    parent_run_id: str | None = None
    goal: str = ""

    def __post_init__(self):
        for item in (self.run_id, self.principal, self.policy_version):
            identifier(item)
        if self.parent_run_id is not None:
            identifier(self.parent_run_id)
        if not math.isfinite(self.expires_at):
            raise ValueError("Expiry must be finite")
        for name in ("tools", "resources", "destinations", "data_classes"):
            values = frozenset(getattr(self, name))
            if any(not isinstance(v, str) or not v or len(v) > 2048 for v in values):
                raise ValueError("Invalid scope")
            object.__setattr__(self, name, values)
        for name in ("max_actions", "max_attempts", "max_cost_units", "max_tokens"):
            nonnegative(getattr(self, name))
        if not isinstance(self.goal, str) or len(self.goal) > 8192:
            raise ValueError("Invalid goal")


@dataclass(frozen=True)
class ProposedAction:
    run_id: str
    action_id: str
    tool: str
    arguments_json: str
    idempotency_key: str

    def __post_init__(self):
        for item in (self.run_id, self.action_id, self.tool, self.idempotency_key):
            identifier(item)
        if (
            not isinstance(self.arguments_json, str)
            or len(self.arguments_json.encode()) > 65536
        ):
            raise ValueError("Arguments exceed 64 KiB")
        value = json.loads(self.arguments_json)
        if type(value) is not dict:
            raise ValueError("Tool arguments must be an object")
        # Reject NaN/Infinity and normalize so equivalent JSON has one digest.
        object.__setattr__(self, "arguments_json", canonical(value))

    @classmethod
    def create(cls, run_id, action_id, tool, arguments, idempotency_key=None):
        return cls(
            run_id, action_id, tool, canonical(arguments), idempotency_key or action_id
        )

    @property
    def arguments(self):
        """Return a fresh copy; mutating caller-owned data cannot change approval."""
        return json.loads(self.arguments_json)

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ActionScope:
    """Derived by trusted adapters from the exact arguments, never the model."""

    resources: frozenset[str]
    destinations: frozenset[str] = frozenset()
    data_classes: frozenset[str] = frozenset({"public"})
    requires_review: bool = False
    cost_units: int = 0
    tokens: int = 0

    def __post_init__(self):
        for name in ("resources", "destinations", "data_classes"):
            values = frozenset(getattr(self, name))
            if any(not isinstance(v, str) or not v or len(v) > 2048 for v in values):
                raise ValueError("Invalid resolved scope")
            object.__setattr__(self, name, values)
        if type(self.requires_review) is not bool:
            raise ValueError("Review requirement must be boolean")
        nonnegative(self.cost_units)
        nonnegative(self.tokens)

    def record(self):
        return {
            **asdict(self),
            "resources": sorted(self.resources),
            "destinations": sorted(self.destinations),
            "data_classes": sorted(self.data_classes),
        }


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str
    action_hash: str
    policy_version: str


@dataclass(frozen=True)
class ActionOutcome:
    decision: Decision
    status: str
    result: Any = field(default=None, repr=False)

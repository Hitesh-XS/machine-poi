"""Host-owned state. Grants and consumed actions live for the service lifetime."""

from dataclasses import dataclass, field

from .contracts import RunState, TaskGrant


@dataclass
class RunRecord:
    grant: TaskGrant
    state: RunState = RunState.RUNNING
    actions: int = 0
    attempts: int = 0
    cost_units: int = 0
    tokens: int = 0
    seen_ids: set[str] = field(default_factory=set)
    seen_keys: set[str] = field(default_factory=set)
    pending: object = None
    tasks: set = field(default_factory=set)
    contexts: set = field(default_factory=set)

"""A review is bound to stored arguments and resolved scope, not new agent text."""

from dataclasses import dataclass

from .contracts import ActionScope, ProposedAction


@dataclass(frozen=True)
class PendingReview:
    action: ProposedAction
    action_hash: str
    expires_at: float
    scope: ActionScope
    tool_version: str

"""Cooperative cancellation contract for trusted async tool adapters."""

import asyncio
from threading import Event


class ExecutionContext:
    def __init__(self, deadline, clock):
        self.deadline = deadline
        self._clock = clock
        self._cancelled = Event()
        self.started = False

    def cancel(self):
        self._cancelled.set()

    def checkpoint(self):
        """Check immediately before each side effect; never swallow cancellation."""
        if self._cancelled.is_set() or self._clock() >= self.deadline:
            raise asyncio.CancelledError

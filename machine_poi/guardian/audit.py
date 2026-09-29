"""Redacted, hash-chained events. Host-owned storage; not external attestation."""

import hashlib
import json
import os
from pathlib import Path
from threading import RLock

from .contracts import canonical


def verify_records(records):
    previous = "0" * 64
    for index, event in enumerate(records):
        body = {key: value for key, value in event.items() if key != "hash"}
        if body.get("sequence") != index or body.get("previous") != previous:
            return False
        expected = hashlib.sha256(canonical(body).encode()).hexdigest()
        if event.get("hash") != expected:
            return False
        previous = expected
    return True


class AuditLog:
    def __init__(self, path=None):
        self._lock = RLock()
        self._records = []
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            self._records = [
                json.loads(line) for line in self._path.read_text().splitlines()
            ]
            if not verify_records(self._records):
                raise ValueError("Invalid audit chain")

    @property
    def records(self):
        with self._lock:
            return json.loads(canonical(self._records))

    def append(self, *, timestamp, run_id, action_hash, event, reason):
        # No arguments, model text, exception messages, credentials or tool results.
        with self._lock:
            body = {
                "sequence": len(self._records),
                "timestamp": timestamp,
                "run_id": run_id,
                "action_hash": action_hash,
                "event": event,
                "reason": reason,
                "previous": self._records[-1]["hash"] if self._records else "0" * 64,
            }
            record = {
                **body,
                "hash": hashlib.sha256(canonical(body).encode()).hexdigest(),
            }
            if self._path:
                descriptor = os.open(
                    self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(canonical(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._records.append(record)

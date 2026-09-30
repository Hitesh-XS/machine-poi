"""Redacted, hash-chained events. Host-owned storage; not external attestation."""

import hashlib
import json
import os
from collections import deque
from pathlib import Path
from threading import RLock

from .contracts import canonical

GENESIS = "0" * 64


def _chain(records, sequence=0, previous=GENESIS):
    """Yield (ok, hash) for each record, continuing a chain from ``previous``."""
    for index, event in enumerate(records, sequence):
        body = {key: value for key, value in event.items() if key != "hash"}
        if body.get("sequence") != index or body.get("previous") != previous:
            yield False, previous
            return
        expected = hashlib.sha256(canonical(body).encode()).hexdigest()
        if event.get("hash") != expected:
            yield False, previous
            return
        previous = expected
        yield True, previous


def verify_records(records, sequence=0, previous=GENESIS):
    """True if ``records`` continue a valid chain (from genesis by default)."""
    return all(ok for ok, _ in _chain(records, sequence, previous))


def _read_file(path, tail):
    """Stream-verify a JSONL chain; return the last ``tail`` records, count and head."""
    kept = deque(maxlen=tail)
    count, previous = 0, GENESIS
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not verify_records([record], count, previous):
                raise ValueError("Invalid audit chain")
            previous = record["hash"]
            count += 1
            kept.append(record)
    return kept, count, previous


class AuditLog:
    """Append-only event chain.

    ``sink`` is a JSONL path (fsynced per event) or a callable that receives
    each canonical record line; a failing sink fails the append. With ``tail``,
    only the last ``tail`` records stay in memory, so a sink is required: the
    sink holds the full chain and ``count``/``head`` continue it.
    """

    def __init__(self, path=None, *, sink=None, tail=None):
        if path is not None and sink is not None:
            raise ValueError("Pass either path or sink")
        sink = path if sink is None else sink
        if tail is not None and (type(tail) is not int or tail < 1):
            raise ValueError("tail must be a positive integer")
        if tail is not None and sink is None:
            raise ValueError("A bounded tail needs a sink that keeps the full chain")
        self._lock = RLock()
        self._path = Path(sink) if isinstance(sink, (str, os.PathLike)) else None
        self._write = sink if callable(sink) else None
        self._records = deque(maxlen=tail)
        self._count = 0
        self._head = GENESIS
        if self._path and self._path.exists():
            self._records, self._count, self._head = _read_file(self._path, tail)

    @property
    def records(self):
        """A copy of the in-memory records (the last ``tail`` when bounded)."""
        with self._lock:
            return json.loads(canonical(list(self._records)))

    @property
    def count(self):
        """Events appended over the chain's lifetime, including a loaded file."""
        with self._lock:
            return self._count

    @property
    def head(self):
        """Hash of the latest event: an anchor for external receipts."""
        with self._lock:
            return self._head

    def verify(self):
        """Verify the full chain in the file sink, or the in-memory records."""
        with self._lock:
            if self._path is not None:
                if not self._path.exists():
                    return self._count == 0
                try:
                    _, count, head = _read_file(self._path, 1)
                except ValueError:
                    return False
                return count == self._count and head == self._head
            if len(self._records) == self._count:
                return verify_records(self._records)
            first = self._records[0] if self._records else None
            return first is None or verify_records(
                self._records, first["sequence"], first["previous"]
            )

    def append(self, *, timestamp, run_id, action_hash, event, reason, count=None):
        # No arguments, model text, exception messages, credentials or tool results.
        with self._lock:
            body = {
                "sequence": self._count,
                "timestamp": timestamp,
                "run_id": run_id,
                "action_hash": action_hash,
                "event": event,
                "reason": reason,
                "previous": self._head,
            }
            if count is not None:
                body["count"] = count
            record = {
                **body,
                "hash": hashlib.sha256(canonical(body).encode()).hexdigest(),
            }
            line = canonical(record)
            if self._path:
                descriptor = os.open(
                    self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            elif self._write is not None:
                self._write(line)
            self._records.append(record)
            self._count += 1
            self._head = record["hash"]

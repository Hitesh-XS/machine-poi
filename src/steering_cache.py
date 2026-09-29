"""Numeric-only steering caches, bound to a model, corpus and recipe.

Metadata prevents accidental reuse; it is not a signature. The host must keep
cache storage outside an untrusted agent's writable filesystem.
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _validated(vectors, hidden_size, num_layers):
    if not vectors:
        raise ValueError("Empty steering cache")
    result = {}
    for layer, value in vectors.items():
        if type(layer) is not int or not 0 <= layer < num_layers:
            raise ValueError("Invalid cached layer")
        array = np.asarray(value)
        if array.shape != (hidden_size,) or array.dtype.kind != "f":
            raise ValueError("Invalid steering vector shape or dtype")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite cached vector")
        result[layer] = array.copy()
    return result


def load_vectors(path, metadata):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Cache exceeds size limit")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("Expanded cache exceeds size limit")
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["metadata"].item())) != metadata:
            raise ValueError("Cache model, revision, corpus or recipe mismatch")
        vectors = {}
        for key in data.files:
            if key == "metadata":
                continue
            if not key.startswith("layer_") or not key[6:].isdigit():
                raise ValueError("Invalid cache key")
            layer = int(key[6:])
            if layer in vectors:
                raise ValueError("Duplicate cache layer")
            vectors[layer] = data[key]
        return _validated(vectors, metadata["hidden_size"], metadata["num_layers"])


def save_vectors(path, vectors, metadata):
    vectors = _validated(vectors, metadata["hidden_size"], metadata["num_layers"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = handle.name
            np.savez(
                handle,
                metadata=np.array(json.dumps(metadata, sort_keys=True)),
                **{f"layer_{k}": v for k, v in vectors.items()},
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)

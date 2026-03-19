# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ObjectStore implementation for MPLang v2 runtime.

ObjectStore provides a two-layer storage model:

- **Transient layer** (MemoryBackend): system-managed in-memory storage for
  runtime values passed between graph executions. Users never interact with
  this layer directly.
- **Persistent layer** (pluggable StoreBackend): user-facing storage for
  checkpointing via ``store.save`` / ``store.load`` dialect. The concrete
  backend (local filesystem, S3, OSS, …) is chosen at deployment time.
"""

from __future__ import annotations

import os
import pickle
import uuid
from abc import ABC, abstractmethod
from typing import Any


class StoreBackend(ABC):
    """Abstract base class for storage backends."""

    @property
    @abstractmethod
    def scheme(self) -> str:
        """URI scheme identifying this backend (e.g. ``mem``, ``fs``, ``s3``)."""
        ...

    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """Store a value with the given key."""
        ...

    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve a value by key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value by key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all keys in the backend."""
        ...


class MemoryBackend(StoreBackend):
    """In-memory storage backend."""

    @property
    def scheme(self) -> str:
        return "mem"

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"Key not found in MemoryBackend: {key}")
        return self._data[key]

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._data

    def list_keys(self) -> list[str]:
        return list(self._data.keys())


class FileSystemBackend(StoreBackend):
    """File system storage backend using pickle."""

    @property
    def scheme(self) -> str:
        return "fs"

    def __init__(self, root_dir: str) -> None:
        self._root = os.path.abspath(root_dir)
        os.makedirs(self._root, exist_ok=True)

    def _get_path(self, key: str) -> str:
        # Security check: prevent directory traversal
        # We assume key is a relative path from root
        # If key starts with /, strip it to make it relative
        clean_key = key.lstrip("/")
        path = os.path.abspath(os.path.join(self._root, clean_key))
        if not path.startswith(self._root):
            raise ValueError(f"Invalid key (traversal attempt): {key}")
        return path

    def put(self, key: str, value: Any) -> None:
        path = self._get_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def get(self, key: str) -> Any:
        path = self._get_path(key)
        if not os.path.exists(path):
            raise KeyError(f"Key not found: {key}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def delete(self, key: str) -> None:
        path = self._get_path(key)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._get_path(key))

    def list_keys(self) -> list[str]:
        keys = []
        for root, _, files in os.walk(self._root):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self._root)
                keys.append(rel_path)
        return keys


class ObjectStore:
    """Object store with scheme-based dispatch.

    Manages two backend layers:

    - **transient** (``mem://``): built-in ``MemoryBackend`` for runtime value
      passing between graph executions.
    - **persistent** (pluggable): optional ``StoreBackend`` for checkpointing
      via the ``store.save`` / ``store.load`` dialect.  The concrete backend
      (filesystem, S3, OSS, ...) is provided at construction time.

    All values are addressed by URIs of the form ``<scheme>://<key>``.
    :meth:`put` / :meth:`get` / :meth:`delete` / :meth:`exists` dispatch to
    the correct backend based on the URI scheme.
    """

    def __init__(self, persistent: StoreBackend | None = None) -> None:
        self._transient = MemoryBackend()
        self._persistent = persistent
        # Build scheme -> backend lookup
        self._backends: dict[str, StoreBackend] = {
            self._transient.scheme: self._transient,
        }
        if persistent is not None:
            if persistent.scheme == self._transient.scheme:
                raise ValueError(
                    f"Persistent backend scheme '{persistent.scheme}' "
                    f"collides with the built-in transient scheme."
                )
            self._backends[persistent.scheme] = persistent

    # ------------------------------------------------------------------
    # URI helpers
    # ------------------------------------------------------------------

    def _parse_uri(self, uri: str) -> tuple[StoreBackend, str]:
        """Parse ``<scheme>://<key>`` and return (backend, key).

        Raises ``ValueError`` for malformed URIs or unknown schemes.
        """
        if "://" not in uri:
            raise ValueError(f"Invalid URI format (missing '://'): {uri}")
        scheme, _, key = uri.partition("://")
        if scheme not in self._backends:
            raise ValueError(
                f"No backend registered for scheme '{scheme}'. "
                f"Available: {list(self._backends)}"
            )
        return self._backends[scheme], key

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def put(self, value: Any, uri: str | None = None) -> str:
        """Store a value.

        Args:
            value: The object to store.
            uri: Optional URI.

                - *None* -> auto-generate ``mem://<uuid>`` (transient).
                - *has scheme* (e.g. ``fs://ckpt/s100``) -> dispatch by scheme;
                  the scheme must match a registered backend.
                - *no scheme* (e.g. ``ckpt/s100``) -> auto-prefix with the
                  persistent backend's scheme.

        Returns:
            The full URI (always includes ``<scheme>://``).
        """
        if uri is None:
            key = uuid.uuid4().hex
            self._transient.put(key, value)
            return f"{self._transient.scheme}://{key}"

        if "://" in uri:
            backend, key = self._parse_uri(uri)
            backend.put(key, value)
            return uri

        # No scheme -> persistent backend
        if self._persistent is None:
            raise RuntimeError(
                "No persistent backend configured. "
                "Pass a StoreBackend to ObjectStore(persistent=...)."
            )
        self._persistent.put(uri, value)
        return f"{self._persistent.scheme}://{uri}"

    def get(self, uri: str) -> Any:
        """Retrieve a value by URI."""
        backend, key = self._parse_uri(uri)
        return backend.get(key)

    def delete(self, uri: str) -> None:
        """Delete a value by URI."""
        backend, key = self._parse_uri(uri)
        backend.delete(key)

    def exists(self, uri: str) -> bool:
        """Check if a URI exists."""
        backend, key = self._parse_uri(uri)
        return backend.exists(key)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def persistent(self) -> StoreBackend | None:
        """Access the persistent backend (for advanced usage or testing)."""
        return self._persistent

    def resolve_uri(self, uri_or_key: str) -> str:
        """Ensure a string is a full URI.

        If *uri_or_key* already contains ``://``, return it unchanged.
        Otherwise, prefix it with the persistent backend's scheme.
        """
        if "://" in uri_or_key:
            return uri_or_key
        if self._persistent is None:
            raise RuntimeError(
                "No persistent backend configured. "
                "Pass a StoreBackend to ObjectStore(persistent=...)."
            )
        return f"{self._persistent.scheme}://{uri_or_key}"

    def list_keys(self) -> list[str]:
        """List all objects as scheme-prefixed URIs.

        Returns a flat list, e.g. ``["mem://abc", "fs://ckpt/s100"]``.
        """
        keys: list[str] = []
        for scheme, backend in self._backends.items():
            keys.extend(f"{scheme}://{k}" for k in backend.list_keys())
        return keys

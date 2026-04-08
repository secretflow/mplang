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
import shutil
import uuid
from abc import ABC, abstractmethod
from hashlib import md5
from typing import Any


class StoreBackend(ABC):
    """Abstract base class for storage backends.

    A backend provides:

    - **Object API** (:meth:`put` / :meth:`get` / :meth:`delete` /
      :meth:`exists`): serialised Python objects.
    - **Data API** (:meth:`download` / :meth:`upload`): file-based
      I/O for table read/write (csv, parquet, …).

    Both APIs resolve paths under a single *root_path*.

    Args:
        root_path: Base path for all storage.  The interpretation is
            backend-specific (local dir, S3 prefix, …).
    """

    def __init__(self, root_path: str | None = None) -> None:
        self._root_path: str | None = root_path

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

    # ------------------------------------------------------------------
    # Data path resolution (for table I/O)
    # ------------------------------------------------------------------

    @abstractmethod
    def download(self, key: str, dest: str) -> None:
        """Download data from the backend to a local path.

        Used by table I/O (csv, parquet, …) as opposed to
        :meth:`put`/:meth:`get` which handle serialized Python objects.

        Implementations should sanitise *key* (strip leading ``/``,
        reject ``..`` segments) and resolve it under :attr:`_root_path`.

        For local backends a file copy (or no-op when paths coincide) is
        performed.  For remote backends (S3, OSS, …) the object is
        downloaded to *dest*.

        Args:
            key: Relative path within the backend's root.
            dest: Local filesystem path to write the data to.
        """
        ...

    @abstractmethod
    def upload(self, source: str, key: str) -> None:
        """Upload data from a local path to the backend.

        For local backends a file copy (or no-op when paths coincide) is
        performed.  For remote backends the local file at *source* is
        uploaded.

        Implementations should sanitise *key* (strip leading ``/``,
        reject ``..`` segments) and resolve it under :attr:`_root_path`.

        Args:
            source: Local filesystem path containing the data.
            key: Relative path within the backend's root.
        """
        ...


class MemoryBackend(StoreBackend):
    """In-memory storage backend."""

    @property
    def scheme(self) -> str:
        return "mem"

    def __init__(self) -> None:
        super().__init__()
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

    def download(self, key: str, dest: str) -> None:
        raise RuntimeError("MemoryBackend does not support file-based data I/O.")

    def upload(self, source: str, key: str) -> None:
        raise RuntimeError("MemoryBackend does not support file-based data I/O.")


class FileSystemBackend(StoreBackend):
    """File system storage backend using pickle."""

    @property
    def scheme(self) -> str:
        return "fs"

    def __init__(self, root_path: str) -> None:
        super().__init__(root_path=root_path)
        os.makedirs(os.path.abspath(root_path), exist_ok=True)

    def _resolve_key(self, key: str) -> str:
        """Resolve *key* under ``root_path`` with traversal prevention."""
        assert self._root_path is not None
        root = os.path.abspath(self._root_path)
        clean_key = key.lstrip("/")
        path = os.path.abspath(os.path.join(root, clean_key))
        if not path.startswith(root):
            raise ValueError(f"Invalid key (traversal attempt): {key}")
        return path

    def put(self, key: str, value: Any) -> None:
        path = self._resolve_key(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def get(self, key: str) -> Any:
        path = self._resolve_key(key)
        if not os.path.exists(path):
            raise KeyError(f"Key not found: {key}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def delete(self, key: str) -> None:
        path = self._resolve_key(key)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._resolve_key(key))

    def list_keys(self) -> list[str]:
        assert self._root_path is not None
        root = os.path.abspath(self._root_path)
        keys = []
        for dirpath, _, files in os.walk(root):
            for file in files:
                full_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(full_path, root)
                keys.append(rel_path)
        return keys

    @staticmethod
    def _file_md5(path: str) -> str:
        hasher = md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _same_file_content(src: str, dest: str) -> bool:
        """Check file equality with a size-first fast path."""
        try:
            return os.path.samefile(src, dest) or (
                os.path.getsize(src) == os.path.getsize(dest)
                and FileSystemBackend._file_md5(src)
                == FileSystemBackend._file_md5(dest)
            )
        except OSError:
            return False

    def download(self, key: str, dest: str) -> None:
        """Materialize backend data at local *dest*.

        - **Absolute *key***: treated as a direct source path.
        - **Relative *key***: resolved under ``root_path``.

        No-op when source and *dest* resolve to the same path.
        If *dest* already exists and is a file/symlink, compare content:
        size differs -> replace directly; size same -> compare MD5;
        same -> no-op and keep existing *dest* as-is,
        different -> replace with symlink to source.

        Raises:
            FileExistsError: If *dest* is an existing directory.
        """
        dest = os.path.abspath(dest)
        if os.path.isabs(key):
            src = os.path.abspath(key)
            if not os.path.exists(src):
                raise FileNotFoundError(
                    f"Download source does not exist: {key!r} (searched {src})"
                )
        else:
            src = self._resolve_key(key)
            if not os.path.exists(src):
                cwd_src = os.path.abspath(key)
                if os.path.exists(cwd_src):
                    src = cwd_src
                else:
                    raise FileNotFoundError(
                        f"Download source does not exist: {key!r} "
                        f"(searched {src} and {cwd_src})"
                    )
        if src != dest:
            if os.path.lexists(dest):
                if os.path.isdir(dest) and not os.path.islink(dest):
                    raise FileExistsError(
                        f"Download destination is an existing directory: {dest}"
                    )
                if self._same_file_content(src, dest):
                    return
                os.remove(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.symlink(src, dest)

    def upload(self, source: str, key: str) -> None:
        """Move data from local *source* to the backend.

        - **Absolute *key***: treated as a direct destination path.
        - **Relative *key***: resolved under ``root_path``.

        No-op when *source* and destination resolve to the same path.
        If destination exists and is a file/symlink, compare content:
        same -> treat as success and remove *source*;
        different -> raise ``FileExistsError`` (no overwrite).
        Uses ``os.rename`` (zero-copy on same filesystem) with
        ``shutil.move`` as fallback for cross-filesystem moves.

        Raises:
            FileExistsError: If destination is an existing directory, or
                destination exists with different content.
        """
        source = os.path.abspath(source)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Upload source does not exist: {source}")
        if os.path.isabs(key):
            dst = os.path.abspath(key)
        else:
            dst = self._resolve_key(key)
        if source != dst:
            if os.path.lexists(dst):
                if os.path.isdir(dst) and not os.path.islink(dst):
                    raise FileExistsError(
                        f"Upload destination is an existing directory: {dst}"
                    )
                if self._same_file_content(source, dst):
                    if os.path.islink(dst):
                        os.remove(dst)
                        shutil.move(source, dst)
                    else:
                        os.remove(source)
                    return
                raise FileExistsError(
                    f"Upload destination already exists with different content: {dst}"
                )
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(source, dst)


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

    # ------------------------------------------------------------------
    # Data path resolution (for table I/O)
    # ------------------------------------------------------------------

    def download(self, path: str, dest: str) -> None:
        """Download / copy a data file to a local destination.

        Delegates to the persistent backend.

        Args:
            path: Data path from the user (stored in IR attrs).
            dest: Local filesystem path to write the data to.
        """
        if self._persistent is None:
            raise RuntimeError(
                "No persistent backend configured. "
                "Pass a StoreBackend to ObjectStore(persistent=...)."
            )
        self._persistent.download(path, dest)

    def upload(self, source: str, path: str) -> None:
        """Upload / copy a local data file to the backend.

        Delegates to the persistent backend.

        Args:
            source: Local filesystem path containing the data.
            path: Data path from the user (stored in IR attrs).
        """
        if self._persistent is None:
            raise RuntimeError(
                "No persistent backend configured. "
                "Pass a StoreBackend to ObjectStore(persistent=...)."
            )
        self._persistent.upload(source, path)

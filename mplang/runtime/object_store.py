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

"""ObjectStore implementation for MPLang v2 runtime."""

from __future__ import annotations

import os
import pickle
import uuid
from abc import ABC, abstractmethod
from typing import Any


class StoreBackend(ABC):
    """Abstract base class for storage backends."""

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
    """Distributed Object Store dispatcher."""

    def __init__(self, fs_root: str) -> None:
        self._backends: dict[str, StoreBackend] = {}
        # Register default memory backend
        self.register_backend("mem", MemoryBackend())
        # Register file system backend
        self.register_backend("fs", FileSystemBackend(root_dir=fs_root))

    def register_backend(self, scheme: str, backend: StoreBackend) -> None:
        """Register a storage backend for a specific URI scheme."""
        self._backends[scheme] = backend

    def _parse_uri(self, uri: str) -> tuple[StoreBackend, str]:
        """Parse URI and return (backend, key)."""
        if "://" not in uri:
            raise ValueError(f"Invalid URI format: {uri}")

        scheme, _, key = uri.partition("://")
        if scheme not in self._backends:
            raise ValueError(f"No backend registered for scheme: {scheme}")

        return self._backends[scheme], key

    def put(self, value: Any, uri: str | None = None) -> str:
        """
        Store a value.

        Args:
            value: The object to store.
            uri: Optional URI. If None, generates 'mem://<uuid>'.

        Returns:
            The URI where the object is stored.
        """
        if uri is None:
            uri = f"mem://{uuid.uuid4()}"

        backend, key = self._parse_uri(uri)
        backend.put(key, value)
        return uri

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

    def list_objects(self) -> list[str]:
        """List all objects in all backends."""
        uris: list[str] = []
        for scheme, backend in self._backends.items():
            try:
                keys = backend.list_keys()
                uris.extend(f"{scheme}://{key}" for key in keys)
            except NotImplementedError:
                pass
        return uris

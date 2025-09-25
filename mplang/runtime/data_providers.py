# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, urlparse

import numpy as np
import pandas as pd

from mplang.backend.base import KernelContext
from mplang.core.table import TableType
from mplang.core.tensor import TensorType
from mplang.utils import table_utils


@dataclass(frozen=True)
class ResolvedURI:
    """Result of resolving a resource path into a normalized form.

    Attributes:
      scheme: The URI scheme (e.g., 'file', 's3', 'mem', 'var', 'secret').
      raw: The original path string as provided by the user.
      parsed: The ParseResult if a scheme was present; otherwise None.
      local_path: For file paths: concrete filesystem path (absolute or as given).
    """

    scheme: str
    raw: str
    parsed: ParseResult | None
    local_path: str | None


def resolve_uri(path: str) -> ResolvedURI:
    """Resolve a user-provided resource location into a normalized URI form.

    This helper accepts plain filesystem paths and RFC 3986 style URIs. A path
    is treated as ``file`` when ``urlparse(path).scheme`` is empty. Detection
    no longer depends on the presence of the literal substring ``"://"`` so
    that forms like ``mem:foo`` (no slashes) are still recognized as a URI.

    Captured fields
    - ``scheme``: Lower-cased scheme (``file`` when absent)
    - ``raw``: Original input
    - ``parsed``: ``ParseResult`` when a scheme was provided, else ``None``
    - ``local_path``: Filesystem path for ``file`` scheme, else ``None``

    Supported (pluggable) schemes out-of-the-box:
      * ``file`` (default)
      * ``mem``
      * ``s3`` (stub)
      * ``secret`` (stub)
      * ``symbols`` (registered server-side)

    Examples
    >>> resolve_uri("data/train.npy").scheme
    'file'
    >>> resolve_uri("mem:dataset1").scheme
    'mem'
    >>> resolve_uri("mem://dataset1").scheme  # both forms acceptable
    'mem'
    >>> resolve_uri("symbols://shared_model").scheme
    'symbols'
    >>> resolve_uri("file:///tmp/x.npy").local_path
    '/tmp/x.npy'
    """

    pr = urlparse(path)
    if not pr.scheme:
        return ResolvedURI("file", path, None, path)

    scheme = pr.scheme.lower()
    local_path: str | None = None
    if scheme == "file":
        local_path = pr.path
        if pr.netloc and not local_path.startswith("/"):
            local_path = f"//{pr.netloc}/{pr.path}"
    return ResolvedURI(scheme, path, pr, local_path)


class DataProvider:
    """Abstract base for data providers.

    Minimal contract: read/write by URI and type spec. Providers may ignore the
    type spec but SHOULD validate when feasible.
    """

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        raise NotImplementedError

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        raise NotImplementedError


_REGISTRY: dict[str, DataProvider] = {}


def register_provider(
    scheme: str, provider: DataProvider, *, replace: bool = False, quiet: bool = False
) -> None:
    """Register a provider implementation.

    Args:
        scheme: URI scheme handled (case-insensitive)
        provider: Implementation
        replace: If False and scheme exists -> ValueError
        quiet: If True, suppress duplicate log messages when replacing
    """
    import logging

    key = scheme.lower()
    if not replace and key in _REGISTRY:
        raise ValueError(f"provider already registered for scheme: {scheme}")
    if replace and key in _REGISTRY and not quiet:
        logging.info(f"Replacing existing provider for scheme '{scheme}'")
    _REGISTRY[key] = provider


def get_provider(scheme: str) -> DataProvider | None:
    return _REGISTRY.get(scheme.lower())


# ---------------- Default Providers ----------------


class FileProvider(DataProvider):
    """Local filesystem provider.

    For tables: CSV bytes via table_utils.
    For tensors: NumPy .npy via np.load/np.save.
    """

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        path = uri.local_path or uri.raw
        if isinstance(out_spec, TableType):
            with open(path, "rb") as f:
                csv_bytes = f.read()
            return table_utils.csv_to_dataframe(csv_bytes)
        # tensor path
        return np.load(path)

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        import os

        path = uri.local_path or uri.raw
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # Table-like to CSV bytes
        if hasattr(value, "__dataframe__") or isinstance(value, pd.DataFrame):
            csv_bytes = table_utils.dataframe_to_csv(value)  # type: ignore
            with open(path, "wb") as f:
                f.write(csv_bytes)
            return
        # Tensor-like via numpy
        np.save(path, np.asarray(value))


class _KeyedPocket:
    """Small helper to keep a dict in KernelContext.state under a namespaced key."""

    def __init__(self, ns: str):
        self.ns = ns

    def get_map(self, ctx: KernelContext) -> dict[str, Any]:
        pocket = ctx.state.setdefault("resource.providers", {})
        store = pocket.get(self.ns)
        if store is None:
            store = {}
            pocket[self.ns] = store
        return store  # type: ignore[return-value]


class MemProvider(DataProvider):
    """In-memory per-runtime KV provider (per rank, per session/runtime)."""

    def __init__(self) -> None:
        self._pocket = _KeyedPocket("mem")

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        store = self._pocket.get_map(ctx)
        key = uri.raw
        if key not in store:
            raise FileNotFoundError(f"mem resource not found: {key}")
        return store[key]

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        store = self._pocket.get_map(ctx)
        store[uri.raw] = value


class S3Provider(DataProvider):
    """Placeholder S3 provider. Install external plugin to enable."""

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        raise NotImplementedError(
            "S3 provider not installed. Provide an external plugin via register_provider('s3', ...) ."
        )

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        raise NotImplementedError(
            "S3 provider not installed. Provide an external plugin via register_provider('s3', ...) ."
        )


class SecretProvider(DataProvider):
    """Placeholder secret provider. Integrate with KMS/secret manager via plugin."""

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        raise NotImplementedError(
            "secret provider not installed. Provide an external plugin via register_provider('secret', ...) ."
        )

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        raise NotImplementedError(
            "secret provider not installed. Provide an external plugin via register_provider('secret', ...) ."
        )


# Register default providers
register_provider("file", FileProvider())
register_provider("mem", MemProvider())
# Stubs to signal missing providers explicitly (can be overridden by plugins)
register_provider("s3", S3Provider())
register_provider("secret", SecretProvider())

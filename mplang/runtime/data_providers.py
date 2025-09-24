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

    This helper accepts both plain filesystem paths and full URIs, returning a
    lightweight ``ResolvedURI`` that captures:
      - ``scheme``: the canonical, lower-cased URI scheme.
      - ``raw``: the original user input (kept verbatim for round-tripping).
      - ``parsed``: the ``urllib.parse.ParseResult`` when a scheme is present; ``None`` otherwise.
      - ``local_path``: a best-effort filesystem path for ``file`` resources; otherwise ``None``.

    Schema (scheme) and standard
    - Syntax follows RFC 3986 (URI Generic Syntax). We detect a URI when the input contains ``"://"``.
    - When no scheme is present, we default to ``file`` and set ``local_path`` to the input string as-is.
    - Supported schemes out-of-the-box (pluggable via ``register_provider``):
        * ``file``: local filesystem paths (default when no scheme is provided)
        * ``mem``: in-memory, per-runtime key-value storage
        * ``var``: per-runtime/session variables
        * ``s3``: placeholder for object storage (requires external plugin)
        * ``secret``: placeholder for secret manager/KMS (requires external plugin)

    Behavior notes for ``file`` scheme
    - ``file:///abs/path`` and ``/abs/path`` are both treated as absolute paths on POSIX.
    - ``file://<host>/abs/path`` will ignore ``<host>`` on POSIX; we keep a path derived from ``ParseResult.path``.
    - ``local_path`` is only populated for the ``file`` scheme; for other schemes it's ``None``.

    Examples
    >>> resolve_uri("data/train.npy")
    ResolvedURI(scheme='file', raw='data/train.npy', parsed=None, local_path='data/train.npy')

    >>> resolve_uri("/abs/path/table.csv")
    ResolvedURI(scheme='file', raw='/abs/path/table.csv', parsed=None, local_path='/abs/path/table.csv')

    >>> resolve_uri("file:///tmp/x.npy").scheme
    'file'

    >>> resolve_uri("mem://ds1").scheme
    'mem'

    >>> resolve_uri("var://session/input").scheme
    'var'

    >>> resolve_uri("s3://bucket/key").scheme
    's3'

    Returns
    -------
    ResolvedURI
        A normalized representation suitable for provider lookups and I/O.
    """

    if "://" not in path:
        return ResolvedURI("file", path, None, path)

    pr = urlparse(path)
    scheme = pr.scheme or "file"
    local_path: str | None = None
    if scheme == "file":
        # file://<host>/abs/path or file:///abs/path; ignore host on posix
        # Keep it simple: join netloc and path when netloc present
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
    scheme: str, provider: DataProvider, *, replace: bool = False
) -> None:
    key = scheme.lower()
    if not replace and key in _REGISTRY:
        raise ValueError(f"provider already registered for scheme: {scheme}")
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


class VarProvider(DataProvider):
    """Session/runtime variable provider.

    Default implementation uses per-runtime state pocket; integration with HTTP
    session symbols can be added by pre-populating this pocket.
    """

    def __init__(self) -> None:
        self._pocket = _KeyedPocket("var")

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        store = self._pocket.get_map(ctx)
        key = uri.raw
        if key not in store:
            raise FileNotFoundError(f"var symbol not found: {key}")
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
register_provider("var", VarProvider())
# Stubs to signal missing providers explicitly (can be overridden by plugins)
register_provider("s3", S3Provider())
register_provider("secret", SecretProvider())

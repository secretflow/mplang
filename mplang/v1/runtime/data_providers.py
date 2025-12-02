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

import pathlib
import struct
from dataclasses import dataclass
from typing import Any
from urllib.parse import ParseResult, urlparse

import numpy as np

from mplang.v1.core import TableLike, TableType, TensorType
from mplang.v1.kernels.base import KernelContext
from mplang.v1.kernels.value import (
    TableValue,
    TensorValue,
    Value,
    decode_value,
    encode_value,
)
from mplang.v1.utils import table_utils


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
MAGIC_MPLANG = b"MPLG"
MAGIC_PARQUET = b"PAR1"
MAGIC_ORC = b"ORC"
MAGIC_NUMPY = b"\x93NUMPY"
VERSION = 0x01


class FileProvider(DataProvider):
    """Local filesystem provider.

    For tables: CSV bytes via table_utils.
    For tensors: NumPy .npy via np.load/np.save.
    """

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        path = pathlib.Path(uri.local_path or uri.raw)
        # try load by magic
        with path.open("rb") as f:
            # this is the maximum length needed to detect all supported formats
            # (numpy requires 6 bytes: '\x93NUMPY').
            MAGIC_BYTES_LEN_MAX = 6
            magic = f.read(MAGIC_BYTES_LEN_MAX)
            f.seek(0)
            if magic.startswith(MAGIC_MPLANG):
                MPLANG_HEADER_LEN = len(MAGIC_MPLANG) + 1
                header = f.read(MPLANG_HEADER_LEN)
                _, version = struct.unpack(">4sB", header)
                if version != VERSION:
                    raise ValueError(f"unsupported mplang version {version}")
                payload = f.read()
                return decode_value(payload)
            elif magic.startswith(MAGIC_PARQUET):
                if not isinstance(out_spec, TableType):
                    raise ValueError(
                        f"PARQUET files require TableType output spec, got {type(out_spec).__name__}"
                    )
                return table_utils.read_table(
                    f, format="parquet", columns=list(out_spec.column_names())
                )
            elif magic.startswith(MAGIC_ORC):
                if not isinstance(out_spec, TableType):
                    raise ValueError(
                        f"ORC files require TableType output spec, got {type(out_spec).__name__}"
                    )
                return table_utils.read_table(
                    f, format="orc", columns=list(out_spec.column_names())
                )
            elif magic.startswith(MAGIC_NUMPY):
                if not isinstance(out_spec, TensorType):
                    raise ValueError(
                        f"NumPy files require TensorType output spec, got {type(out_spec).__name__}"
                    )
                return np.load(f)

            # Fallback: open the file for CSV or NumPy loading.
            if isinstance(out_spec, TableType):
                return table_utils.read_table(
                    f, format="csv", columns=list(out_spec.column_names())
                )
            else:
                return np.load(f)

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        import os

        path = uri.local_path or uri.raw
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if not isinstance(value, Value):
            value = (
                TableValue(value)
                if isinstance(value, TableLike)
                else TensorValue(value)
            )

        if isinstance(value, TableValue):
            table_utils.write_table(value.to_arrow(), path, format="parquet")
        elif isinstance(value, TensorValue):
            with open(path, "wb") as f:
                np.save(f, value.to_numpy())
        else:
            payload = encode_value(value)
            with open(path, "wb") as f:
                f.write(struct.pack(">4sB", MAGIC_MPLANG, VERSION))
                f.write(payload)


class MemProvider(DataProvider):
    """In-memory per-runtime KV provider (per rank, per session/runtime)."""

    STATE_KEY = "resource.providers.mem"

    @staticmethod
    def _store(ctx: KernelContext) -> dict[str, Any]:
        # Use ensure_state so creation is atomic & centralized; enforce dict.
        store = ctx.runtime.ensure_state(MemProvider.STATE_KEY, dict)
        if not isinstance(store, dict):  # pragma: no cover - defensive
            raise TypeError(
                f"runtime state key '{MemProvider.STATE_KEY}' expected dict, got {type(store).__name__}"
            )
        return store  # type: ignore[return-value]

    def read(
        self, uri: ResolvedURI, out_spec: TensorType | TableType, *, ctx: KernelContext
    ) -> Any:
        store = self._store(ctx)
        key = uri.raw
        if key not in store:
            raise FileNotFoundError(f"mem resource not found: {key}")
        return store[key]

    def write(self, uri: ResolvedURI, value: Any, *, ctx: KernelContext) -> None:
        store = self._store(ctx)
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

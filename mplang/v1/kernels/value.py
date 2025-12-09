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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from mplang.v1.protos.v1alpha1 import value_pb2 as _value_pb2

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "BytesBlob",
    "TableValue",
    "TensorValue",
    "Value",
    "ValueDecodeError",
    "ValueError",
    "ValueProtoBuilder",
    "ValueProtoReader",
    "decode_value",
    "encode_value",
    "is_value_envelope",
    "list_value_kinds",
    "register_value",
]


class ValueError(Exception):  # shadow built-in intentionally local
    """Base exception for backend Value related errors."""


class ValueDecodeError(ValueError):
    """Raised when decoding a Value envelope or payload fails."""


class ValueProtoBuilder:
    """Builder for creating ValueProto messages with a fluent API.

    Provides a cleaner, more ergonomic interface than directly working with pb2 messages.

    Example:
        proto = ValueProtoBuilder("my.custom.Type", version=1) \
            .set_attr("shape", [1, 2, 3]) \
            .set_attr("dtype", "float32") \
            .set_payload(b"some bytes") \
            .build()
    """

    def __init__(self, kind: str, version: int):
        """Initialize a new proto builder.

        Args:
            kind: The globally unique KIND identifier for this Value type
            version: The WIRE_VERSION for this Value type
        """
        self._proto = _value_pb2.ValueProto()
        self._proto.kind = kind
        self._proto.value_version = version

    def set_payload(self, payload: bytes) -> ValueProtoBuilder:
        """Set the payload bytes.

        Args:
            payload: Raw bytes to store in the proto

        Returns:
            Self for method chaining
        """
        self._proto.payload = payload
        return self

    def set_attr(self, key: str, value: Any) -> ValueProtoBuilder:
        """Set a single runtime attribute.

        Args:
            key: Attribute key
            value: Attribute value. Supported types: bool, int, float, str, bytes,
                   list[int/float/str]

        Returns:
            Self for method chaining
        """
        self._proto.runtime_attrs[key].CopyFrom(_python_to_attr_proto(value))
        return self

    def build(self) -> _value_pb2.ValueProto:
        """Return the built proto.

        Returns:
            The fully constructed ValueProto
        """
        return self._proto


class ValueProtoReader:
    """Reader for extracting data from ValueProto messages.

    Provides a convenient interface for reading proto fields and attributes.

    Example:
        reader = ValueProtoReader(proto)
        shape = reader.get_attr("shape")
        dtype = reader.get_attr("dtype")
        payload = reader.payload
    """

    def __init__(self, proto: _value_pb2.ValueProto):
        """Initialize a reader for an existing proto.

        Args:
            proto: The ValueProto to read from
        """
        self._proto = proto

    @property
    def kind(self) -> str:
        """Get the KIND identifier."""
        return self._proto.kind

    @property
    def version(self) -> int:
        """Get the WIRE_VERSION."""
        return self._proto.value_version

    @property
    def payload(self) -> bytes:
        """Get the payload bytes."""
        return self._proto.payload

    def get_attr(self, key: str, default: Any = ...) -> Any:
        """Get a single attribute with optional default.

        Args:
            key: Attribute key to retrieve
            default: Default value if key is missing. If not provided (default),
                     raises ValueDecodeError when key is missing.

        Returns:
            The decoded attribute value or default

        Raises:
            ValueDecodeError: If key is missing and no default provided
        """
        if key not in self._proto.runtime_attrs:
            if default is ...:
                raise ValueDecodeError(f"Missing required runtime_attr: {key}")
            return default
        return _attr_proto_to_python(self._proto.runtime_attrs[key])


class Value(ABC):
    """Abstract base for backend-level transferable values.

    Subclasses MUST define:
        KIND (ClassVar[str]): globally unique stable identifier
        WIRE_VERSION (ClassVar[int]): per-kind payload version integer >=1

    Use ValueProtoBuilder and ValueProtoReader for proto serialization.
    """

    KIND: ClassVar[str]
    WIRE_VERSION: ClassVar[int] = 1

    def estimated_wire_size(self) -> int | None:  # optional hint
        return None

    @abstractmethod
    def to_proto(self) -> _value_pb2.ValueProto:
        """Return fully-populated ValueProto for this value."""

    @classmethod
    @abstractmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> Value:
        """Construct instance from a parsed ValueProto."""

    def to_bool(self) -> bool:
        """Convert value to bool (for predicates in control flow).

        Default implementation raises NotImplementedError.
        Subclasses should override to provide appropriate conversion.

        Raises:
            NotImplementedError: If the value type cannot be converted to bool
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support conversion to bool"
        )


T = TypeVar("T", bound=Value)

_VALUE_REGISTRY: dict[str, type[Value]] = {}


def _python_to_attr_proto(value: Any) -> _value_pb2.ValueAttrProto:
    attr = _value_pb2.ValueAttrProto()
    if isinstance(value, bool):
        attr.type = _value_pb2.ValueAttrProto.BOOL
        attr.b = value
    elif isinstance(value, int) and not isinstance(value, bool):
        attr.type = _value_pb2.ValueAttrProto.INT
        attr.i = value
    elif isinstance(value, float):
        attr.type = _value_pb2.ValueAttrProto.FLOAT
        attr.f = value
    elif isinstance(value, str):
        attr.type = _value_pb2.ValueAttrProto.STRING
        attr.s = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        attr.type = _value_pb2.ValueAttrProto.BYTES
        attr.raw_bytes = bytes(value)
    elif isinstance(value, (list, tuple)):
        if not value:
            # Represent empty list explicitly
            attr.type = _value_pb2.ValueAttrProto.EMPTY
            return attr
        if all(isinstance(v, float) for v in value):
            attr.type = _value_pb2.ValueAttrProto.FLOATS
            attr.floats.extend(float(v) for v in value)
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            attr.type = _value_pb2.ValueAttrProto.INTS
            attr.ints.extend(int(v) for v in value)
        elif all(isinstance(v, str) for v in value):
            attr.type = _value_pb2.ValueAttrProto.STRINGS
            attr.strs.extend(value)
        else:
            raise TypeError(
                "Unsupported iterable element type for AttrProto: "
                f"{type(value[0]).__name__}"
            )
    elif value is None:
        attr.type = _value_pb2.ValueAttrProto.UNDEFINED
    else:
        raise TypeError(
            "Unsupported runtime attr type for Value serialization: "
            f"{type(value).__name__}"
        )
    return attr


def _attr_proto_to_python(attr: _value_pb2.ValueAttrProto) -> Any:
    if attr.type == _value_pb2.ValueAttrProto.FLOAT:
        return attr.f
    if attr.type == _value_pb2.ValueAttrProto.INT:
        return attr.i
    if attr.type == _value_pb2.ValueAttrProto.STRING:
        return attr.s
    if attr.type == _value_pb2.ValueAttrProto.BOOL:
        return attr.b
    if attr.type == _value_pb2.ValueAttrProto.BYTES:
        return attr.raw_bytes
    if attr.type == _value_pb2.ValueAttrProto.FLOATS:
        return list(attr.floats)
    if attr.type == _value_pb2.ValueAttrProto.INTS:
        return list(attr.ints)
    if attr.type == _value_pb2.ValueAttrProto.STRINGS:
        return list(attr.strs)
    if attr.type == _value_pb2.ValueAttrProto.EMPTY:
        return []
    if attr.type == _value_pb2.ValueAttrProto.UNDEFINED:
        return None
    raise ValueDecodeError(f"Unsupported AttrProto type {attr.type}")


def _looks_like_pyarrow_table(obj: Any) -> bool:
    if obj is None:
        return False
    module = getattr(obj.__class__, "__module__", "")
    return (
        module.startswith("pyarrow.")
        and hasattr(obj, "schema")
        and hasattr(obj, "column_names")
        and hasattr(obj, "num_rows")
    )


def _looks_like_pandas_df(obj: Any) -> bool:
    if obj is None:
        return False
    module = getattr(obj.__class__, "__module__", "")
    return (
        module.startswith("pandas.")
        and hasattr(obj, "columns")
        and hasattr(obj, "dtypes")
    )


def register_value(cls: type[T]) -> type[T]:
    kind = getattr(cls, "KIND", None)
    if not kind or not isinstance(kind, str):
        raise ValueError(f"Value subclass {cls.__name__} missing KIND str")
    if kind in _VALUE_REGISTRY:
        raise ValueError(f"Duplicate Value KIND '{kind}'")
    if getattr(cls, "WIRE_VERSION", None) is None:
        raise ValueError(f"Value subclass {cls.__name__} missing WIRE_VERSION")
    _VALUE_REGISTRY[kind] = cls
    return cls


def list_value_kinds() -> list[str]:
    return sorted(_VALUE_REGISTRY.keys())


def encode_value(val: Value) -> bytes:
    """Encode using protobuf envelope.

    Raises:
        ValueError if protobuf module not available.
    """
    if _value_pb2 is None:  # pragma: no cover
        raise ValueError("protobuf value_pb2 not generated yet")
    proto = val.to_proto()
    if not isinstance(proto, _value_pb2.ValueProto):
        raise ValueError("Value.to_proto must return ValueProto")
    if not proto.kind:
        proto.kind = val.KIND
    elif proto.kind != val.KIND:
        raise ValueError(
            f"ValueProto.kind mismatch: expected '{val.KIND}', got '{proto.kind}'"
        )
    if proto.value_version == 0:
        proto.value_version = val.WIRE_VERSION
    elif proto.value_version != val.WIRE_VERSION:
        raise ValueError(
            f"ValueProto.value_version mismatch: expected {val.WIRE_VERSION}, got {proto.value_version}"
        )
    return proto.SerializeToString()  # type: ignore[no-any-return]


def is_value_envelope(data: bytes) -> bool:
    if _value_pb2 is None:
        return False
    env = _value_pb2.ValueProto()
    try:
        env.ParseFromString(data)
        return bool(env.kind)
    except Exception:
        return False


def decode_value(data: bytes) -> Value:
    if _value_pb2 is None:
        raise ValueDecodeError("protobuf value_pb2 not available for decode")
    env = _value_pb2.ValueProto()
    try:
        env.ParseFromString(data)
    except Exception as e:  # pragma: no cover
        raise ValueDecodeError(f"Failed parsing ValueProto: {e}") from e
    if not env.kind:
        raise ValueDecodeError("Envelope missing kind")
    cls = _VALUE_REGISTRY.get(env.kind)
    if cls is None:
        raise ValueDecodeError(f"Unknown Value kind '{env.kind}'")
    if env.value_version and env.value_version != cls.WIRE_VERSION:
        raise ValueDecodeError(
            f"Unsupported {cls.__name__} version {env.value_version}"
        )
    if env.value_version == 0:
        env.value_version = cls.WIRE_VERSION
    return cls.from_proto(env)


@register_value
class BytesBlob(Value):  # demo subclass
    KIND = "mplang.demo.BytesBlob"
    WIRE_VERSION = 1

    def __init__(self, data: bytes):
        self._data = data

    def to_proto(self) -> _value_pb2.ValueProto:
        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_payload(self._data)
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> BytesBlob:
        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported BytesBlob version {reader.version}")
        if proto.runtime_attrs:
            raise ValueDecodeError("BytesBlob does not expect runtime attributes")
        return cls(reader.payload)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BytesBlob(len={len(self._data)})"


@register_value
class TensorValue(Value):  # well-known tensor (ndarray) Value
    """Numpy ndarray serialization via raw buffer + runtime metadata."""

    KIND = "mplang.ndarray"
    WIRE_VERSION = 1

    def __init__(self, array):  # type: ignore[no-untyped-def]
        import numpy as np

        if not isinstance(array, np.ndarray):
            raise TypeError("TensorValue expects a numpy.ndarray")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self._arr = array

    def to_proto(self) -> _value_pb2.ValueProto:
        import numpy as np

        arr = self._arr
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_attr("dtype", arr.dtype.str)
            .set_attr("shape", [int(dim) for dim in arr.shape])
            .set_payload(arr.tobytes())
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> TensorValue:
        import numpy as np

        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported TensorValue version {reader.version}")
        dtype_val = reader.get_attr("dtype")
        if not isinstance(dtype_val, str):
            raise ValueDecodeError("TensorValue runtime attr 'dtype' must be str")
        shape_val = reader.get_attr("shape")
        if not isinstance(shape_val, list):
            raise ValueDecodeError("TensorValue runtime attr 'shape' must be list")
        shape = tuple(int(dim) for dim in shape_val)
        try:
            arr = np.frombuffer(reader.payload, dtype=np.dtype(dtype_val)).reshape(
                shape
            )
        except Exception as e:  # pragma: no cover
            raise ValueDecodeError(f"Failed reconstruct ndarray: {e}") from e
        return cls(np.array(arr, copy=True))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype[Any]:  # pragma: no cover - simple accessor
        return self._arr.dtype  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:  # pragma: no cover - simple accessor
        return int(self._arr.ndim)

    def to_numpy(
        self, *, copy: bool = False
    ) -> np.ndarray[Any, Any]:  # pragma: no cover - simple accessor
        if copy:
            import numpy as np

            return np.array(self._arr, copy=True)
        return self._arr  # type: ignore[no-any-return]

    def __array__(
        self, dtype: np.dtype[Any] | None = None
    ) -> np.ndarray[Any, Any]:  # pragma: no cover - numpy bridge
        import numpy as np

        return np.asarray(self._arr, dtype=dtype)

    def to_bool(self) -> bool:
        """Convert tensor to bool (for scalar predicates).

        Returns:
            bool value if tensor is scalar

        Raises:
            ValueError: If tensor is not a scalar
        """
        if self._arr.size != 1:
            raise ValueError(
                f"Cannot convert non-scalar tensor (shape={self._arr.shape}) to bool"
            )
        return bool(self._arr.item())

    def __repr__(self) -> str:  # pragma: no cover
        return f"TensorValue(shape={self._arr.shape}, dtype={self._arr.dtype})"


@register_value
class TableValue(Value):  # well-known table (Arrow IPC) Value
    """Table value backed by PyArrow, serialized via Arrow IPC stream.

    KIND: mplang.dataframe.arrow
    WIRE_VERSION: increments if wire semantics (not DataFrame contents) change.

    Internal representation: Always pyarrow.Table for consistency and performance.
    Wire format: Arrow IPC stream.

    Accepts pandas DataFrame or pyarrow.Table as input, but internally converts
    everything to pyarrow.Table for unified handling.
    """

    KIND = "mplang.dataframe.arrow"
    WIRE_VERSION = 1

    def __init__(self, data):  # type: ignore[no-untyped-def]
        """Initialize TableValue from pandas DataFrame or pyarrow.Table.

        Args:
            data: pandas.DataFrame, pyarrow.Table, or dict-like object
        """
        try:
            import pyarrow as pa  # type: ignore
        except ImportError as e:
            raise ValueError("pyarrow is required for TableValue") from e

        if _looks_like_pyarrow_table(data):
            self._table = data
        elif _looks_like_pandas_df(data):
            try:
                self._table = pa.Table.from_pandas(data, preserve_index=False)
            except Exception as e:
                raise ValueError(
                    f"Cannot convert pandas DataFrame to Arrow: {e}"
                ) from e
        else:
            # Try to convert dict-like or other structures
            try:
                self._table = pa.table(data)
            except Exception as e:
                raise TypeError(
                    f"TableValue requires pandas.DataFrame or pyarrow.Table, got {type(data).__name__}"
                ) from e

    def to_proto(self) -> _value_pb2.ValueProto:
        """Serialize to Arrow IPC stream format."""
        import pyarrow as pa  # type: ignore
        import pyarrow.ipc as pa_ipc  # type: ignore

        sink = pa.BufferOutputStream()
        with pa_ipc.new_stream(sink, self._table.schema) as writer:  # type: ignore[arg-type]
            writer.write_table(self._table)  # type: ignore[arg-type]
        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_payload(sink.getvalue().to_pybytes())
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> TableValue:
        """Deserialize from Arrow IPC stream format."""
        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported TableValue version {reader.version}")
        if proto.runtime_attrs:
            raise ValueDecodeError("TableValue does not expect runtime attributes")

        import pyarrow as pa  # type: ignore
        import pyarrow.ipc as pa_ipc  # type: ignore

        buf = pa.py_buffer(reader.payload)
        ipc_reader = pa_ipc.open_stream(buf)
        table = ipc_reader.read_all()
        return cls(table)

    def to_arrow(self) -> Any:  # pyarrow.Table
        """Get the underlying pyarrow.Table (primary interface).

        Returns:
            pyarrow.Table: The table data
        """
        return self._table

    def to_pandas(self) -> Any:  # pandas.DataFrame
        """Convert to pandas DataFrame (compatibility interface).

        Note: This creates a copy and converts from Arrow to pandas format.
        For better performance, consider using to_arrow() and working with
        Arrow-native APIs (DuckDB, etc.) directly.

        Returns:
            pandas.DataFrame: Converted dataframe
        """
        return self._table.to_pandas()  # type: ignore[attr-defined]

    @property
    def columns(self) -> list[str]:
        """Return column names (TableLike protocol compatibility)."""
        return [str(name) for name in self._table.column_names]

    @property
    def dtypes(self) -> Any:  # pyarrow.Schema
        """Return column dtypes as Arrow schema (TableLike protocol compatibility)."""
        return self._table.schema

    def num_rows(self) -> int:
        """Get number of rows in the table.

        Returns:
            Number of rows
        """
        return self._table.num_rows  # type: ignore[attr-defined,return-value,no-any-return]

    def __repr__(self) -> str:
        """String representation of TableValue."""
        try:
            rows = self.num_rows()
            cols = self.columns
            return f"TableValue(rows={rows}, cols={cols})"
        except Exception:
            return "TableValue()"

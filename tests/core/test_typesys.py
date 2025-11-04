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

"""Tests for the class-based type system (mplang.core.typesys).

This suite validates:
- Base type constructors (TensorType, TableType)
- Representation types (EncodedType, EncryptedType, SecretSharedType)
- Multi-party type (MpType)
- Minimal predicates (is_tensor, unwrap_repr, same_dtype, etc.)
- Type composition and immutability
"""

import pytest

from mplang.core.dtypes import FLOAT32, FLOAT64, INT64, STRING
from mplang.core.mask import Mask
from mplang.core.typesys import (
    EncodedType,
    EncryptedType,
    MpType,
    SecretSharedType,
    TableType,
    TensorType,
    is_mp,
    is_table,
    is_tensor,
    same_dtype,
    same_schema,
    same_security,
    same_shape,
    unwrap_repr,
)


class TestTensorType:
    """Test TensorType construction, validation, and repr."""

    def test_basic_construction(self):
        t = TensorType(FLOAT32, (3, 4))
        assert is_tensor(t)
        assert not is_table(t)
        assert t.dtype == FLOAT32
        assert t.shape == (3, 4)
        assert repr(t) == "f32[3, 4]"

    def test_dynamic_dims(self):
        t = TensorType(INT64, (-1, 10))
        assert t.shape == (-1, 10)
        assert "i64[-1, 10]" in repr(t)

    def test_empty_shape(self):
        t = TensorType(FLOAT32, ())
        assert t.shape == ()
        assert repr(t) == "f32"

    def test_invalid_dtype(self):
        with pytest.raises(TypeError, match="dtype must be DType"):
            TensorType("i64", (1,))  # type: ignore[arg-type]

    def test_invalid_shape_dims(self):
        with pytest.raises(TypeError, match="dims must all be ints"):
            TensorType(INT64, (1, "bad"))  # type: ignore[arg-type]


class TestTableType:
    """Test TableType construction, validation, and repr."""

    def test_basic_construction(self):
        t = TableType((("id", INT64), ("name", STRING)))
        assert is_table(t)
        assert not is_tensor(t)
        assert t.columns == (("id", INT64), ("name", STRING))
        assert repr(t) == "Tbl(id:i64, name:str)"

    def test_empty_schema_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            TableType(())

    def test_duplicate_column_names_rejected(self):
        with pytest.raises(ValueError, match="names must be unique"):
            TableType((("id", INT64), ("id", STRING)))

    def test_empty_column_name_rejected(self):
        with pytest.raises(ValueError, match="names cannot be empty"):
            TableType((("", INT64),))

    def test_invalid_column_dtype(self):
        with pytest.raises(TypeError, match="type must be DType"):
            TableType((("id", "int64"),))  # type: ignore[arg-type]


class TestEncodedType:
    """Test EncodedType (representation wrapper)."""

    def test_basic_construction(self):
        base = TensorType(INT64, (1024,))
        et = EncodedType(base, codec="fixed", params={"scale": 16})
        assert et.inner == base
        assert et.codec == "fixed"
        assert et.params == (("scale", 16),)
        assert not is_tensor(et)
        # Verify we can unwrap to base
        assert unwrap_repr(et) == base

    def test_repr(self):
        base = TensorType(INT64, (1024,))
        et = EncodedType(base, codec="fixed", params={"scale": 16})
        r = repr(et)
        assert "Enc" in r and "fixed" in r and "scale" in r

    def test_params_normalization(self):
        base = TensorType(INT64, (10,))
        et1 = EncodedType(base, codec="fp", params={"a": 1, "b": 2})
        et2 = EncodedType(base, codec="fp", params={"b": 2, "a": 1})
        # params are normalized to sorted tuple
        assert et1.params == et2.params
        assert et1 == et2


class TestEncryptedType:
    """Test EncryptedType (HE wrapper)."""

    def test_basic_construction(self):
        base = TensorType(FLOAT64, (2048,))
        ht = EncryptedType(base, scheme="ckks", params={"n": 16384})
        assert ht.inner == base
        assert ht.scheme == "ckks"
        assert ht.params == (("n", 16384),)
        # Verify we can unwrap to base
        assert unwrap_repr(ht) == base

    def test_key_ref_optional(self):
        base = TensorType(FLOAT64, (128,))
        ht = EncryptedType(base, scheme="bfv", key_ref="key123")
        assert ht.key_ref == "key123"
        assert "key=key123" in repr(ht) or "key123" in repr(ht)


class TestSecretSharedType:
    """Test SecretSharedType (MPC/SS wrapper)."""

    def test_basic_construction(self):
        base = TensorType(INT64, (10, 10))
        st = SecretSharedType(base, scheme="aby3", field_bits=64)
        assert st.inner == base
        assert st.scheme == "aby3"
        assert st.field_bits == 64
        # Verify we can unwrap to base
        assert unwrap_repr(st) == base

    def test_repr(self):
        base = TensorType(INT64, (10, 10))
        st = SecretSharedType(base, scheme="aby3", field_bits=64)
        r = repr(st)
        assert "SS" in r and "aby3" in r and "field_bits" in r


class TestMpType:
    """Test MpType (multi-party distribution type)."""

    def test_basic_construction(self):
        base = TensorType(INT64, (10,))
        pmask = Mask.from_int(0b111)
        mp = MpType(base, pmask=pmask)
        assert is_mp(mp)
        assert not is_tensor(mp)
        assert mp.inner == base
        assert mp.pmask == pmask
        assert mp.quals is None

    def test_with_quals(self):
        base = TableType((("id", INT64), ("val", FLOAT32)))
        pmask = Mask.from_int(0xF)
        mp = MpType(base, pmask=pmask, quals={"device": "P0", "exec": "gpu"})
        assert mp.quals == {"device": "P0", "exec": "gpu"}
        assert "device" in repr(mp)

    def test_no_pmask(self):
        base = TensorType(FLOAT32, (3, 3))
        mp = MpType(base, pmask=None)
        assert mp.pmask is None
        assert "Mp(" in repr(mp)

    def test_invalid_inner(self):
        with pytest.raises(TypeError, match="inner must be a Type"):
            MpType("not a type", pmask=None)  # type: ignore[arg-type]

    def test_unwrap_with_mptype(self):
        base = TensorType(INT64, (5,))
        encoded = EncodedType(base, codec="fp")
        mp = MpType(encoded, pmask=Mask.from_int(0b11))
        # Unwrap MpType then representation wrappers
        inner = mp.inner
        unwrapped = unwrap_repr(inner)
        assert unwrapped == base


class TestWrapperComposition:
    """Test composing multiple wrapper types (e.g., Encrypted(Encoded(Tensor)))."""

    def test_enc_then_he(self):
        base = TensorType(INT64, (100,))
        encoded = EncodedType(base, codec="fixed", params={"scale": 8})
        encrypted = EncryptedType(encoded, scheme="ckks", params={"n": 8192})
        # Unwrap all representation wrappers
        unwrapped = unwrap_repr(encrypted)
        assert unwrapped == base
        assert isinstance(unwrapped, TensorType)

    def test_ss_then_enc(self):
        base = TableType((("id", INT64), ("val", FLOAT32)))
        shared = SecretSharedType(base, scheme="aby3")
        encoded = EncodedType(shared, codec="zkp")
        # Unwrap all representation wrappers
        unwrapped = unwrap_repr(encoded)
        assert unwrapped == base
        assert isinstance(unwrapped, TableType)


class TestPredicates:
    """Test minimal predicates and utilities."""

    def test_unwrap_repr(self):
        base = TensorType(INT64, (5,))
        enc1 = EncodedType(base, codec="fp")
        enc2 = EncryptedType(enc1, scheme="bfv")
        enc3 = SecretSharedType(enc2, scheme="3pc")
        # unwrap_repr strips all repr wrappers
        unwrapped = unwrap_repr(enc3)
        assert unwrapped == base

    def test_same_dtype(self):
        t1 = TensorType(INT64, (10, 20))
        t2 = TensorType(INT64, (5, 5))
        t3 = TensorType(FLOAT32, (10, 20))
        assert same_dtype(t1, t2)
        assert not same_dtype(t1, t3)
        # Works through wrappers
        e1 = EncodedType(t1, codec="fp")
        e2 = SecretSharedType(t2, scheme="aby3")
        assert same_dtype(e1, e2)

    def test_same_shape(self):
        t1 = TensorType(INT64, (10, 20))
        t2 = TensorType(FLOAT32, (10, 20))
        t3 = TensorType(INT64, (5, 5))
        assert same_shape(t1, t2)
        assert not same_shape(t1, t3)

    def test_same_schema(self):
        tab1 = TableType((("a", INT64), ("b", STRING)))
        tab2 = TableType((("a", INT64), ("b", STRING)))
        tab3 = TableType((("x", INT64), ("y", STRING)))
        assert same_schema(tab1, tab2)
        assert not same_schema(tab1, tab3)

    def test_same_security(self):
        base = TensorType(INT64, (10,))
        e1 = EncodedType(base, codec="fixed", params={"scale": 16})
        h1 = EncryptedType(e1, scheme="ckks", params={"n": 8192})
        e2 = EncodedType(base, codec="fixed", params={"scale": 16})
        h2 = EncryptedType(e2, scheme="ckks", params={"n": 8192})
        assert same_security(h1, h2)
        # Different params
        e3 = EncodedType(base, codec="fixed", params={"scale": 8})
        h3 = EncryptedType(e3, scheme="ckks", params={"n": 8192})
        assert not same_security(h1, h3)

    def test_unwrap_base_invalid(self):
        # Can't wrap non-Type
        with pytest.raises(TypeError):
            EncodedType("not a type", codec="fp")  # type: ignore[arg-type]


class TestImmutability:
    """Ensure all types are frozen/immutable."""

    def test_tensor_frozen(self):
        t = TensorType(INT64, (10,))
        with pytest.raises(Exception):  # dataclass frozen=True
            t.dtype = FLOAT32  # type: ignore[misc]

    def test_wrapper_frozen(self):
        base = TensorType(INT64, (10,))
        et = EncodedType(base, codec="fp")
        with pytest.raises(Exception):
            et.codec = "other"  # type: ignore[misc]

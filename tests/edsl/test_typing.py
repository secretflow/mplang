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

"""
Tests for the MPLang typing system.

This test suite validates the design principles outlined in mplang/core/typing.py:
1. Orthogonality and Composition: Tests that Layout, Encryption, and Distribution types compose correctly.
2. The "Three Worlds" of HE: Tests the strict separation between plaintext, element-wise HE, and SIMD HE.
3. Contracts via Protocols: Tests that ScalarType hierarchy and EncryptedTrait behave as expected.
"""

import pytest

from mplang2.edsl.typing import (
    HE,
    MP,
    SIMD_HE,
    SS,
    BaseType,
    Custom,
    CustomType,
    EncryptedTrait,
    ScalarHEType,
    ScalarType,
    SIMDHEType,
    SSType,
    Table,
    TableType,
    Tensor,
    TensorType,
    f32,
    f64,
    i32,
    i64,
)

# ==============================================================================
# --- Test Scalar Types (Pillar 1: Layout Types)
# ==============================================================================


class TestScalarType:
    """Test ScalarType definitions and behavior."""

    def test_scalar_types_exist(self):
        """Test that predefined scalar types are available."""
        assert str(f32) == "f32"
        assert str(f64) == "f64"
        assert str(i32) == "i32"
        assert str(i64) == "i64"

    def test_scalar_type_str_representation(self):
        """Test string representation of scalar types."""
        assert str(f32) == "f32"
        assert str(i64) == "i64"

    def test_scalar_type_is_base_type(self):
        """Test that ScalarType inherits from BaseType."""
        assert isinstance(f32, BaseType)
        assert isinstance(i32, BaseType)

    def test_scalar_type_inheritance(self):
        """Test that scalar types inherit from ScalarType."""
        assert isinstance(f32, ScalarType)
        assert isinstance(i64, ScalarType)

    def test_custom_scalar_type(self):
        """Test that IntegerType can be used to create custom bit-width integers."""
        from mplang2.edsl.typing import IntegerType

        custom = IntegerType(bitwidth=1, signed=False)  # bool-like type
        assert str(custom) == "u1"
        assert isinstance(custom, ScalarType)


# ==============================================================================
# --- Test Tensor Types (Pillar 1: Layout Types)
# ==============================================================================


class TestTensorType:
    """Test TensorType construction and validation."""

    def test_tensor_construction_basic(self):
        """Test basic tensor construction with element type and shape."""
        t = TensorType(f32, (10, 20))
        assert t.element_type == f32
        assert t.shape == (10, 20)

    def test_tensor_class_getitem_syntax(self):
        """Test the Tensor[elem_type, shape] syntax."""
        t = Tensor[f32, (5, 5)]
        assert isinstance(t, TensorType)
        assert t.element_type == f32
        assert t.shape == (5, 5)

    def test_tensor_str_representation(self):
        """Test string representation of tensors."""
        t = Tensor[i32, (100,)]
        assert str(t) == "Tensor[i32, (100)]"

        t2 = Tensor[f64, (3, 4, 5)]
        assert str(t2) == "Tensor[f64, (3, 4, 5)]"

    def test_tensor_requires_scalar_element(self):
        """Test that Tensor requires ScalarType element types."""
        # Valid: ScalarType instances
        t = Tensor[f32, (10,)]
        assert t.element_type == f32

        # Valid: HE[scalar] is also a ScalarType (ScalarHEType inherits ScalarType)
        t_he = Tensor[HE[f64], (20,)]
        assert isinstance(t_he.element_type, ScalarType)

    def test_tensor_rejects_non_scalar_element(self):
        """Test that Tensor rejects non-ScalarType element types (e.g., SIMD_HE)."""
        simd_he = SIMD_HE[f32, (4096,)]
        with pytest.raises(TypeError, match="Tensor element type must be a ScalarType"):
            Tensor[simd_he, (4,)]

    def test_tensor_empty_shape(self):
        """Test tensor with empty shape (scalar-like)."""
        t = Tensor[f32, ()]
        assert t.shape == ()
        assert str(t) == "Tensor[f32, ()]"

    def test_tensor_1d_shape(self):
        """Test 1D tensor."""
        t = Tensor[i64, (1000,)]
        assert t.shape == (1000,)

    def test_tensor_scalar_representation(self):
        """Test Tensor[i32, ()] - scalar as 0-dim tensor."""
        t = Tensor[i32, ()]
        assert t.element_type == i32
        assert t.shape == ()
        assert str(t) == "Tensor[i32, ()]"

    def test_tensor_partial_dynamic_dim(self):
        """Test Tensor[i32, (-1, 10)] - partially dynamic shape."""
        t = Tensor[i32, (-1, 10)]
        assert t.element_type == i32
        assert t.shape == (-1, 10)
        assert str(t) == "Tensor[i32, (-1, 10)]"

    def test_tensor_fully_ranked(self):
        """Test Tensor[i32, (3, 10)] - fully static/ranked tensor."""
        t = Tensor[i32, (3, 10)]
        assert t.element_type == i32
        assert t.shape == (3, 10)
        assert str(t) == "Tensor[i32, (3, 10)]"

    def test_tensor_dynamic_shape_1d(self):
        """Test 1D tensor with dynamic shape."""
        t = Tensor[f64, (-1,)]
        assert t.element_type == f64
        assert t.shape == (-1,)
        assert str(t) == "Tensor[f64, (-1)]"

    def test_tensor_dynamic_shape_both_dims(self):
        """Test 2D tensor with both dimensions dynamic."""
        t = Tensor[i32, (-1, -1)]
        assert t.shape == (-1, -1)

    def test_tensor_invalid_dimension_zero(self):
        """Test that zero dimensions are invalid."""
        with pytest.raises(ValueError, match="Invalid dimension 0"):
            Tensor[f32, (0, 10)]

    def test_tensor_invalid_dimension_negative(self):
        """Test that negative dimensions other than -1 are invalid."""
        with pytest.raises(ValueError, match="Invalid dimension -2"):
            Tensor[f32, (-2, 10)]

    def test_tensor_invalid_dimension_type(self):
        """Test that non-integer dimensions are rejected."""
        with pytest.raises(TypeError, match="Shape dimensions must be integers"):
            Tensor[f32, (10.5, 20)]

    def test_tensor_requires_shape_parameter(self):
        """Test that Tensor requires explicit shape parameter."""
        with pytest.raises(TypeError, match="Tensor requires shape parameter"):
            Tensor[f32]  # Missing shape parameter

    def test_tensor_equality(self):
        """Test tensor type equality."""
        t1 = Tensor[f32, (10, 20)]
        t2 = Tensor[f32, (10, 20)]
        t3 = Tensor[f32, (10, 30)]
        t4 = Tensor[i32, (10, 20)]

        assert t1 == t2
        assert t1 != t3  # Different shape
        assert t1 != t4  # Different dtype

    def test_tensor_hash(self):
        """Test tensor type hashing for use in sets/dicts."""
        t1 = Tensor[f32, (10, 20)]
        t2 = Tensor[f32, (10, 20)]
        t3 = Tensor[f32, (10, 30)]

        # Equal tensors should have same hash
        assert hash(t1) == hash(t2)

        # Can be used in sets
        tensor_set = {t1, t2, t3}
        assert len(tensor_set) == 2  # t1 and t2 are the same

    def test_tensor_is_scalar_property(self):
        """Test is_scalar property."""
        assert Tensor[f32, ()].is_scalar
        assert not Tensor[f32, (1,)].is_scalar
        assert not Tensor[f32, (10, 20)].is_scalar

    def test_tensor_is_fully_static_property(self):
        """Test is_fully_static property."""
        assert Tensor[f32, (3, 10)].is_fully_static
        assert Tensor[f32, ()].is_fully_static  # Scalar is fully static
        assert not Tensor[f32, (-1, 10)].is_fully_static
        assert not Tensor[f32, (-1, -1)].is_fully_static

    def test_tensor_rank_property(self):
        """Test rank property."""
        assert Tensor[f32, ()].rank == 0  # Scalar
        assert Tensor[f32, (10,)].rank == 1
        assert Tensor[f32, (3, 10)].rank == 2
        assert Tensor[f32, (-1, 10, 5)].rank == 3  # Dynamic dims still have known rank

    def test_tensor_has_dynamic_dims(self):
        """Test has_dynamic_dims method."""
        assert Tensor[f32, (-1, 10)].has_dynamic_dims()
        assert Tensor[f32, (-1, -1)].has_dynamic_dims()
        assert not Tensor[f32, (3, 10)].has_dynamic_dims()
        assert not Tensor[f32, ()].has_dynamic_dims()  # Scalar has no dynamic dims
        assert not Tensor[f32, ()].has_dynamic_dims()


# ==============================================================================
# --- Test Table Types (Pillar 1: Layout Types)
# ==============================================================================


class TestTableType:
    """Test TableType construction and schema handling."""

    def test_table_construction(self):
        """Test basic table construction with schema."""
        schema = {"age": i32, "score": f32}
        t = TableType(schema)
        assert t.schema == schema

    def test_table_class_getitem_syntax(self):
        """Test the Table[{...}] syntax."""
        t = Table[{"col_a": i32, "col_b": f64}]
        assert isinstance(t, TableType)
        assert "col_a" in t.schema
        assert t.schema["col_a"] == i32

    def test_table_str_representation(self):
        """Test string representation of tables."""
        t = Table[{"x": f32, "y": i64}]
        s = str(t)
        assert "Table[" in s
        assert "'x': f32" in s
        assert "'y': i64" in s

    def test_table_empty_schema(self):
        """Test table with empty schema."""
        t = Table[{}]
        assert t.schema == {}
        assert str(t) == "Table[{}]"

    def test_table_complex_schema(self):
        """Test table with complex types in schema."""
        schema = {
            "tensor_col": Tensor[f32, (10,)],
            "scalar_col": i32,
        }
        t = Table[schema]
        assert isinstance(t.schema["tensor_col"], TensorType)
        assert t.schema["scalar_col"] == i32


# ==============================================================================
# --- Test Custom Opaque Types
# ==============================================================================


class TestCustomType:
    """Test CustomType for domain-specific opaque types."""

    def test_custom_type_construction(self):
        """Test basic CustomType construction."""
        key_type = CustomType("crypto.encryption_key")
        assert key_type.kind == "crypto.encryption_key"

    def test_custom_type_subscript_syntax(self):
        """Test Custom[kind] subscript syntax."""
        key_type = Custom["crypto.key"]
        assert isinstance(key_type, CustomType)
        assert key_type.kind == "crypto.key"

    def test_custom_type_str_representation(self):
        """Test string representation of CustomType."""
        key_type = Custom["crypto.key"]
        assert str(key_type) == "Custom[crypto.key]"

        handle = Custom["tee.handle"]
        assert str(handle) == "Custom[tee.handle]"

    def test_custom_type_repr(self):
        """Test repr of CustomType."""
        key_type = CustomType("MyKey")
        assert repr(key_type) == "CustomType('MyKey')"

    def test_custom_type_equality(self):
        """Test CustomType equality based on kind."""
        key1 = Custom["crypto.key"]
        key2 = Custom["crypto.key"]
        key3 = Custom["crypto.other_key"]

        assert key1 == key2
        assert key1 != key3

    def test_custom_type_hash(self):
        """Test CustomType can be used in sets and dicts."""
        key1 = Custom["crypto.key"]
        key2 = Custom["crypto.key"]
        key3 = Custom["tee.handle"]

        # Same kind should have same hash
        assert hash(key1) == hash(key2)

        # Can be used in sets
        type_set = {key1, key2, key3}
        assert len(type_set) == 2  # key1 and key2 are equal

        # Can be used as dict keys
        type_dict = {key1: "value1", key3: "value3"}
        assert len(type_dict) == 2

    def test_custom_type_invalid_kind_type(self):
        """Test that kind must be a string."""
        with pytest.raises(TypeError, match="kind must be str"):
            CustomType(123)

    def test_custom_type_invalid_kind_empty(self):
        """Test that kind must be non-empty."""
        with pytest.raises(ValueError, match="kind must be a non-empty string"):
            CustomType("")

        with pytest.raises(ValueError, match="kind must be a non-empty string"):
            CustomType("   ")

    def test_custom_type_is_base_type(self):
        """Test that CustomType inherits from BaseType."""
        key_type = Custom["crypto.key"]
        assert isinstance(key_type, BaseType)

    def test_custom_type_not_scalar(self):
        """Test that CustomType is NOT a ScalarType."""
        key_type = Custom["crypto.key"]
        assert not isinstance(key_type, ScalarType)

    def test_custom_type_not_encrypted_trait(self):
        """Test that CustomType does NOT implement EncryptedTrait."""
        key_type = Custom["crypto.key"]
        assert not isinstance(key_type, EncryptedTrait)

    def test_custom_type_use_cases(self):
        """Test realistic CustomType use cases."""
        # Cryptographic key
        enc_key = Custom["crypto.encryption_key"]
        dec_key = Custom["crypto.decryption_key"]
        assert enc_key != dec_key

        # TEE handles
        tee_handle = Custom["tee.secure_handle"]
        tee_context = Custom["tee.context"]
        assert tee_handle != tee_context

        # Database handles
        db_conn = Custom["database.connection"]
        db_cursor = Custom["database.cursor"]
        assert db_conn != db_cursor

    def test_custom_type_dotted_notation(self):
        """Test that dotted notation works for namespacing."""
        # Domain.subtype notation
        t1 = Custom["crypto.he.key"]
        t2 = Custom["crypto.phe.key"]
        t3 = Custom["tee.sgx.handle"]

        assert t1.kind == "crypto.he.key"
        assert t2.kind == "crypto.phe.key"
        assert t3.kind == "tee.sgx.handle"

        # All different
        assert len({t1, t2, t3}) == 3


# ==============================================================================
# --- Test HE Types (Pillar 2: Encryption Types)
# ==============================================================================


class TestScalarHEType:
    """Test ScalarHEType (HE) construction and behavior."""

    def test_he_construction_basic(self):
        """Test basic HE construction."""
        he = ScalarHEType()
        assert he._scheme == "ckks"

    def test_he_class_getitem_syntax(self):
        """Test the HE[scheme] syntax."""
        he = HE["ckks"]
        assert isinstance(he, ScalarHEType)
        assert he._scheme == "ckks"

    def test_he_str_representation(self):
        """Test string representation of HE types."""
        he = HE["ckks"]
        assert str(he) == "HE[ckks]"

    def test_he_is_scalar_type(self):
        """Test that HE inherits from ScalarType."""
        he = HE["ckks"]
        assert isinstance(he, ScalarType)

    def test_he_is_encrypted_trait(self):
        """Test that HE implements EncryptedTrait."""
        he = HE["ckks"]
        assert isinstance(he, EncryptedTrait)

    def test_he_custom_scheme(self):
        """Test HE with custom scheme parameter."""
        he = ScalarHEType(scheme="bfv")
        assert he._scheme == "bfv"

    def test_he_getitem_with_scheme(self):
        """Test HE[scheme] syntax."""
        he = HE["bfv"]
        assert he._scheme == "bfv"


# ==============================================================================
# --- Test Encryption Types (Pillar 2: SIMD HE)
# ==============================================================================


class TestSIMDHEType:
    """Test SIMDHEType (SIMD_HE) construction and the 'World 3' constraints."""

    def test_simd_he_construction_basic(self):
        """Test basic SIMD_HE construction."""
        simd = SIMDHEType(f32, (4096,))
        assert simd.scalar_type == f32
        assert simd.packing_shape == (4096,)
        assert simd._scheme == "ckks"

    def test_simd_he_class_getitem_syntax(self):
        """Test the SIMD_HE[scalar, shape] syntax."""
        simd = SIMD_HE[f32, (8192,)]
        assert isinstance(simd, SIMDHEType)
        assert simd.scalar_type == f32
        assert simd.packing_shape == (8192,)

    def test_simd_he_str_representation(self):
        """Test string representation of SIMD_HE types."""
        simd = SIMD_HE[f64, (2048,)]
        assert str(simd) == "SIMD_HE[f64, (2048,)]"

    def test_simd_he_pt_type_is_tensor(self):
        """Test that SIMD_HE.pt_type returns a TensorType."""
        simd = SIMD_HE[f32, (1024,)]
        pt = simd.pt_type
        assert isinstance(pt, TensorType)
        assert pt.element_type == f32
        assert pt.shape == (1024,)

    def test_simd_he_is_encrypted_trait(self):
        """Test that SIMD_HE implements EncryptedTrait."""
        simd = SIMD_HE[f32, (512,)]
        assert isinstance(simd, EncryptedTrait)
        pt_type = simd.pt_type
        assert isinstance(pt_type, TensorType)

    def test_simd_he_not_scalar_type(self):
        """Test that SIMD_HE is NOT a ScalarType (World 3 constraint)."""
        simd = SIMD_HE[f32, (4096,)]
        assert not isinstance(simd, ScalarType)

    def test_simd_he_cannot_be_tensor_element(self):
        """Test that SIMD_HE cannot be an element of a Tensor (World 3 constraint)."""
        simd = SIMD_HE[f32, (4096,)]
        with pytest.raises(TypeError, match="Tensor element type must be a ScalarType"):
            Tensor[simd, (10,)]

    def test_simd_he_requires_scalar_type(self):
        """Test that SIMD_HE requires a ScalarType."""
        with pytest.raises(TypeError, match="SIMD_HE requires a ScalarType"):
            SIMD_HE[Tensor[f32, (10,)], (100,)]

    def test_simd_he_custom_scheme(self):
        """Test SIMD_HE with custom scheme."""
        simd = SIMDHEType(f64, (2048,), scheme="bfv")
        assert simd._scheme == "bfv"


# ==============================================================================
# --- Test Encryption Types (Pillar 2: Secret Sharing)
# ==============================================================================


class TestSSType:
    """Test SSType (SS) construction and secret sharing semantics."""

    def test_ss_construction_basic(self):
        """Test basic SS construction."""
        ss = SSType(f32)
        assert ss.pt_type == f32
        assert ss._enc_schema == "ss"

    def test_ss_class_getitem_syntax(self):
        """Test the SS[type] syntax."""
        ss = SS[i32]
        assert isinstance(ss, SSType)
        assert ss.pt_type == i32

    def test_ss_str_representation(self):
        """Test string representation of SS types."""
        ss = SS[f64]
        assert str(ss) == "SS[f64]"

    def test_ss_is_encrypted_trait(self):
        """Test that SS implements EncryptedTrait."""
        ss = SS[f32]
        assert isinstance(ss, EncryptedTrait)
        assert ss.pt_type == f32
        assert ss.enc_schema == "ss"

    def test_ss_of_tensor(self):
        """Test secret sharing of a tensor."""
        tensor_type = Tensor[f32, (10, 10)]
        ss = SS[tensor_type]
        assert isinstance(ss.pt_type, TensorType)
        assert ss.pt_type.element_type == f32
        assert ss.pt_type.shape == (10, 10)

    def test_ss_of_table(self):
        """Test secret sharing of a table."""
        table_type = Table[{"x": i32, "y": f64}]
        ss = SS[table_type]
        assert isinstance(ss.pt_type, TableType)

    def test_ss_custom_schema(self):
        """Test SS with custom encryption schema."""
        ss = SSType(f32, enc_schema="replicated_ss")
        assert ss.enc_schema == "replicated_ss"


# ==============================================================================
# --- Test Distribution Types (Pillar 3)
# ==============================================================================


class TestMPType:
    """Test MPType (MP) for multi-party distribution."""

    def test_mp_construction_basic(self):
        """Test basic MP construction."""
        from mplang2.edsl.typing import MPType

        mp = MPType(f32, (0, 1))
        assert mp.value_type == f32
        assert mp.parties == (0, 1)

    def test_mp_class_getitem_syntax(self):
        """Test the MP[type, parties] syntax."""
        mp = MP[f32, (0, 1, 2)]
        assert mp.value_type == f32
        assert mp.parties == (0, 1, 2)

    def test_mp_str_representation(self):
        """Test string representation of MP types."""
        mp = MP[i32, (0,)]
        s = str(mp)
        assert "MP[" in s
        assert "i32" in s
        assert "parties=(0,)" in s

    def test_mp_of_tensor(self):
        """Test multi-party distribution of a tensor."""
        tensor = Tensor[f64, (100,)]
        mp = MP[tensor, (0, 1)]
        assert isinstance(mp.value_type, TensorType)
        assert mp.parties == (0, 1)

    def test_mp_of_encrypted_type(self):
        """Test multi-party distribution of encrypted types."""
        he = HE[f32]
        mp = MP[he, (2, 3)]
        assert isinstance(mp.value_type, ScalarHEType)
        assert mp.parties == (2, 3)


# ==============================================================================
# --- Test Type Composition (Principle 1: Orthogonality)
# ==============================================================================


class TestTypeComposition:
    """Test composition of Layout, Encryption, and Distribution types."""

    def test_composition_world_1_plaintext(self):
        """Test World 1: Plaintext Tensor."""
        plain_tensor = Tensor[f32, (10, 20)]
        assert isinstance(plain_tensor, TensorType)
        assert plain_tensor.element_type == f32
        assert plain_tensor.shape == (10, 20)

    def test_composition_world_2_elementwise_he(self):
        """Test World 2: Element-wise HE Tensor."""
        # HE is a ScalarType, so it can be a Tensor element
        he_tensor = Tensor[HE["ckks"], (100,)]
        assert isinstance(he_tensor, TensorType)
        assert isinstance(he_tensor.element_type, ScalarHEType)

    def test_composition_world_3_simd_he(self):
        """Test World 3: SIMD HE (opaque, non-tensor)."""
        simd_he = SIMD_HE[f32, (4096,)]
        assert isinstance(simd_he, SIMDHEType)
        assert not isinstance(simd_he, ScalarType)

    def test_composition_ss_tensor(self):
        """Test secret sharing of a tensor."""
        tensor = Tensor[i32, (5, 5)]
        ss_tensor = SS[tensor]
        assert isinstance(ss_tensor, SSType)
        assert isinstance(ss_tensor.pt_type, TensorType)

    def test_composition_mp_ss_tensor(self):
        """Test MP[SS[Tensor[...]]] composition."""
        tensor = Tensor[f32, (10,)]
        ss_tensor = SS[tensor]
        mp_ss_tensor = MP[ss_tensor, (0, 1)]

        assert isinstance(mp_ss_tensor.value_type, SSType)
        assert isinstance(mp_ss_tensor.value_type.pt_type, TensorType)
        assert mp_ss_tensor.parties == (0, 1)

    def test_composition_mp_simd_he(self):
        """Test MP[SIMD_HE[...]] composition."""
        simd_he = SIMD_HE[f32, (8192,)]
        mp_simd = MP[simd_he, (2,)]

        assert isinstance(mp_simd.value_type, SIMDHEType)
        assert mp_simd.parties == (2,)

    def test_composition_mp_he_tensor(self):
        """Test MP[Tensor[HE[...], ...]] composition."""
        he_tensor = Tensor[HE["ckks"], (20,)]
        mp_he_tensor = MP[he_tensor, (0, 1, 2)]

        assert isinstance(mp_he_tensor.value_type, TensorType)
        assert isinstance(mp_he_tensor.value_type.element_type, ScalarHEType)
        assert mp_he_tensor.parties == (0, 1, 2)

    def test_composition_table_with_mixed_types(self):
        """Test table with mixed column types."""
        schema = {
            "plain": f32,
            "tensor": Tensor[i32, (10,)],
            "encrypted": HE["ckks"],
        }
        table = Table[schema]
        assert isinstance(table.schema["plain"], ScalarType)
        assert isinstance(table.schema["tensor"], TensorType)
        assert isinstance(table.schema["encrypted"], ScalarHEType)


# ==============================================================================
# --- Test Protocol Contracts (Principle 3)
# ==============================================================================


class TestProtocolContracts:
    """Test that types satisfy the expected protocol contracts."""

    def test_scalar_type_hierarchy(self):
        """Test that ScalarType hierarchy is implemented correctly."""
        # Scalar types are ScalarType instances
        assert isinstance(f32, ScalarType)
        assert isinstance(i64, ScalarType)

        # HE[scalar] inherits from ScalarType
        assert isinstance(HE[f32], ScalarType)

        # SIMD_HE does NOT inherit from ScalarType
        assert not isinstance(SIMD_HE[f32, (1024,)], ScalarType)

    def test_encrypted_trait_protocol(self):
        """Test that EncryptedTrait is implemented correctly."""

        # HE implements EncryptedTrait
        he = HE["ckks"]
        assert isinstance(he, EncryptedTrait)

        # SIMD_HE implements EncryptedTrait
        simd = SIMD_HE[f64, (2048,)]
        assert isinstance(simd, EncryptedTrait)
        assert isinstance(simd.pt_type, TensorType)

        # SS implements EncryptedTrait
        ss = SS[i32]
        assert isinstance(ss, EncryptedTrait)
        assert ss.pt_type == i32
        assert ss.enc_schema == "ss"

    def test_base_type_inheritance(self):
        """Test that all type classes inherit from BaseType."""
        assert isinstance(f32, BaseType)
        assert isinstance(Tensor[f32, (10,)], BaseType)
        assert isinstance(Table[{"x": i32}], BaseType)
        assert isinstance(HE[f32], BaseType)
        assert isinstance(SIMD_HE[f32, (1024,)], BaseType)
        assert isinstance(SS[f32], BaseType)
        assert isinstance(MP[f32, (0,)], BaseType)

    def test_repr_uses_str(self):
        """Test that __repr__ delegates to __str__."""
        t = Tensor[f32, (5,)]
        assert repr(t) == str(t)


# ==============================================================================
# --- Test Edge Cases and Error Handling
# ==============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling in the typing system."""

    def test_tensor_with_invalid_element_type(self):
        """Test that Tensor rejects invalid element types."""
        # MP is not a ScalarType
        mp_type = MP[f32, (0,)]
        with pytest.raises(TypeError, match="Tensor element type must be a ScalarType"):
            Tensor[mp_type, (10,)]

        # Table is not a ScalarType
        table_type = Table[{"x": i32}]
        with pytest.raises(TypeError, match="Tensor element type must be a ScalarType"):
            Tensor[table_type, (5,)]

    def test_simd_he_with_non_scalar_plaintext(self):
        """Test that SIMD_HE requires ScalarType plaintext."""
        tensor = Tensor[f32, (10,)]
        with pytest.raises(TypeError, match="SIMD_HE requires a ScalarType"):
            SIMD_HE[tensor, (1024,)]

    def test_nested_encryption_allowed(self):
        """Test that nested encryption is allowed (though may be unusual)."""
        # SS[HE[f32]] - a share of an encrypted value
        nested = SS[HE[f32]]
        assert isinstance(nested, SSType)
        assert isinstance(nested.pt_type, ScalarHEType)

    def test_double_distribution_allowed(self):
        """Test that double distribution is allowed (though may be unusual)."""
        # MP[MP[f32, (0,)], (1,)]
        inner_mp = MP[f32, (0,)]
        outer_mp = MP[inner_mp, (1,)]
        assert isinstance(outer_mp.value_type, type(inner_mp))


# ==============================================================================
# --- Test Demonstration Cases (from module __main__)
# ==============================================================================


class TestDemonstrationCases:
    """Test the demonstration cases from the typing module's __main__ block."""

    def test_world_1_demonstration(self):
        """Test World 1: Plaintext tensor demonstration."""
        plain_tensor = Tensor[f32, (10, 20)]
        assert str(plain_tensor) == "Tensor[f32, (10, 20)]"

    def test_world_2_demonstration(self):
        """Test World 2: Element-wise HE tensor demonstration."""
        elementwise_he_tensor = Tensor[HE["ckks"], (100,)]
        assert "Tensor[HE[ckks]" in str(elementwise_he_tensor)
        assert "(100)" in str(elementwise_he_tensor)

    def test_world_3_demonstration(self):
        """Test World 3: SIMD HE vector demonstration."""
        simd_he_vector = SIMD_HE[f32, (4096,)]
        assert str(simd_he_vector) == "SIMD_HE[f32, (4096,)]"

    def test_design_constraint_demonstration(self):
        """Test that SIMD_HE cannot be a Tensor element (design constraint)."""
        simd_he_vector = SIMD_HE[f32, (4096,)]
        with pytest.raises(TypeError, match="Tensor element type must be a ScalarType"):
            Tensor[simd_he_vector, (4,)]

    def test_secret_sharing_demonstration(self):
        """Test secret sharing demonstration."""
        ss_tensor_share = SS[Tensor[i32, (5, 5)]]
        assert "SS[Tensor[i32" in str(ss_tensor_share)

    def test_composition_with_distribution_demonstration(self):
        """Test composition with distribution demonstration."""
        ss_tensor_share = SS[Tensor[i32, (5, 5)]]
        mp_ss_tensor = MP[ss_tensor_share, (0, 1)]
        assert "MP[SS[Tensor" in str(mp_ss_tensor)

        simd_he_vector = SIMD_HE[f32, (4096,)]
        mp_simd_he_vector = MP[simd_he_vector, (2,)]
        assert "MP[SIMD_HE" in str(mp_simd_he_vector)
        assert "parties=(2,)" in str(mp_simd_he_vector)


# ==============================================================================
# --- Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for complex type composition scenarios."""

    def test_realistic_mpc_scenario(self):
        """Test a realistic MPC scenario: MP[SS[Tensor[f32, ...]]]."""
        # A tensor of floats, secret-shared among parties 0, 1, 2
        tensor = Tensor[f32, (100, 100)]
        ss_tensor = SS[tensor]
        mp_ss_tensor = MP[ss_tensor, (0, 1, 2)]

        # Verify the structure
        assert isinstance(mp_ss_tensor, type(MP[f32, (0,)]))
        assert isinstance(mp_ss_tensor.value_type, SSType)
        assert isinstance(mp_ss_tensor.value_type.pt_type, TensorType)
        assert mp_ss_tensor.value_type.pt_type.element_type == f32
        assert mp_ss_tensor.value_type.pt_type.shape == (100, 100)
        assert mp_ss_tensor.parties == (0, 1, 2)

    def test_realistic_he_scenario(self):
        """Test a realistic HE scenario: MP[Tensor[HE[...], ...]]."""
        # A tensor of element-wise encrypted values held by party 0
        he_tensor = Tensor[HE["ckks"], (50,)]
        mp_he_tensor = MP[he_tensor, (0,)]

        assert isinstance(mp_he_tensor.value_type, TensorType)
        assert isinstance(mp_he_tensor.value_type.element_type, ScalarHEType)
        assert mp_he_tensor.parties == (0,)

    def test_realistic_simd_he_scenario(self):
        """Test a realistic SIMD HE scenario: MP[SIMD_HE[...]]."""
        # A SIMD ciphertext held by multiple parties
        simd = SIMD_HE[f32, (8192,)]
        mp_simd = MP[simd, (1, 2)]

        assert isinstance(mp_simd.value_type, SIMDHEType)
        assert mp_simd.value_type.scalar_type == f32
        assert mp_simd.value_type.packing_shape == (8192,)
        assert mp_simd.parties == (1, 2)

    def test_table_in_distributed_setting(self):
        """Test a table distributed among parties."""
        schema = {
            "id": i64,
            "features": Tensor[f32, (10,)],
            "label": i32,
        }
        table = Table[schema]
        mp_table = MP[table, (0, 1)]

        assert isinstance(mp_table.value_type, TableType)
        assert mp_table.parties == (0, 1)
        assert isinstance(mp_table.value_type.schema["features"], TensorType)

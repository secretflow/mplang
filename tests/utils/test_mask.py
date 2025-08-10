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

import pytest

# Direct import to avoid mplang init dependencies
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mplang', 'utils'))
import mask

from mask import (
    Mask,
    bit_count,
    ensure_rank_in,
    ensure_subset,
    enum_mask,
    global_to_relative_rank,
    is_rank_in,
    is_subset,
    relative_to_global_rank,
)


class TestMaskClass:
    def test_mask_creation_from_int(self):
        m = Mask(5)
        assert m.value == 5
        assert int(m) == 5

    def test_mask_creation_from_mask(self):
        m1 = Mask(5)
        m2 = Mask(m1)
        assert m2.value == 5
        assert m1 == m2

    def test_mask_negative_value_raises(self):
        with pytest.raises(ValueError):
            Mask(-1)

    def test_mask_invalid_type_raises(self):
        with pytest.raises(TypeError):
            Mask("invalid")

    def test_mask_bit_count(self):
        assert Mask(0).bit_count() == 0
        assert Mask(1).bit_count() == 1
        assert Mask(3).bit_count() == 2
        assert Mask(7).bit_count() == 3
        assert Mask(15).bit_count() == 4

    def test_mask_enum(self):
        assert list(Mask(0).enum()) == []
        assert list(Mask(1).enum()) == [0]
        assert list(Mask(5).enum()) == [0, 2]
        assert list(Mask(7).enum()) == [0, 1, 2]

    def test_mask_is_rank_in(self):
        m = Mask(5)  # 0b101
        assert m.is_rank_in(0) is True
        assert m.is_rank_in(1) is False
        assert m.is_rank_in(2) is True

    def test_mask_contains(self):
        m = Mask(5)  # 0b101
        assert 0 in m
        assert 1 not in m
        assert 2 in m

    def test_mask_ensure_rank_in(self):
        m = Mask(5)  # 0b101
        m.ensure_rank_in(0)  # Should not raise
        m.ensure_rank_in(2)  # Should not raise
        with pytest.raises(ValueError):
            m.ensure_rank_in(1)

    def test_mask_is_subset(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        assert subset.is_subset(superset) is True
        assert subset.is_subset(subset) is True
        assert superset.is_subset(subset) is False

    def test_mask_ensure_subset(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        subset.ensure_subset(superset)  # Should not raise
        with pytest.raises(ValueError):
            superset.ensure_subset(subset)

    def test_mask_global_to_relative_rank(self):
        m = Mask(11)  # 0b1011 (parties 0, 1, 3)
        assert m.global_to_relative_rank(0) == 0
        assert m.global_to_relative_rank(1) == 1
        assert m.global_to_relative_rank(3) == 2

    def test_mask_relative_to_global_rank(self):
        m = Mask(11)  # 0b1011 (parties 0, 1, 3)
        assert m.relative_to_global_rank(0) == 0
        assert m.relative_to_global_rank(1) == 1
        assert m.relative_to_global_rank(2) == 3

    def test_mask_bitwise_operations(self):
        m1 = Mask(5)  # 0b101
        m2 = Mask(3)  # 0b011
        
        assert (m1 & m2).value == 1  # 0b001
        assert (m1 | m2).value == 7  # 0b111
        assert (m1 ^ m2).value == 6  # 0b110
        
        # Test with integers
        assert (m1 & 3).value == 1
        assert (m1 | 2).value == 7

    def test_mask_shift_operations(self):
        m = Mask(5)  # 0b101
        assert (m << 1).value == 10  # 0b1010
        assert (m >> 1).value == 2   # 0b010

    def test_mask_comparison_operations(self):
        m1 = Mask(5)
        m2 = Mask(3)
        m3 = Mask(5)
        
        assert m1 == m3
        assert m1 != m2
        assert m1 > m2
        assert m2 < m1
        assert m1 >= m3
        assert m2 <= m1
        
        # Test with integers
        assert m1 == 5
        assert m1 != 3

    def test_mask_hash(self):
        m1 = Mask(5)
        m2 = Mask(5)
        m3 = Mask(3)
        
        assert hash(m1) == hash(m2)
        assert hash(m1) != hash(m3)
        
        # Can be used in sets
        mask_set = {m1, m2, m3}
        assert len(mask_set) == 2

    def test_mask_repr_and_str(self):
        m = Mask(5)
        assert repr(m) == "Mask(5)"
        assert str(m) == "5"


class TestBitCount:
    def test_bit_count_zero(self):
        assert bit_count(0) == 0

    def test_bit_count_single_bit(self):
        assert bit_count(1) == 1
        assert bit_count(2) == 1
        assert bit_count(4) == 1
        assert bit_count(8) == 1

    def test_bit_count_multiple_bits(self):
        assert bit_count(3) == 2  # 0b11
        assert bit_count(7) == 3  # 0b111
        assert bit_count(15) == 4  # 0b1111
        assert bit_count(5) == 2  # 0b101

    def test_bit_count_large_numbers(self):
        assert bit_count(255) == 8  # 0b11111111
        assert bit_count(1023) == 10  # 0b1111111111

    def test_bit_count_with_mask_object(self):
        assert bit_count(Mask(5)) == 2


class TestEnumMask:
    def test_enum_mask_zero(self):
        result = list(enum_mask(0))
        assert result == []

    def test_enum_mask_single_bit(self):
        result = list(enum_mask(1))  # 0b1
        assert result == [0]

        result = list(enum_mask(4))  # 0b100
        assert result == [2]

    def test_enum_mask_multiple_bits(self):
        result = list(enum_mask(5))  # 0b101
        assert result == [0, 2]

        result = list(enum_mask(7))  # 0b111
        assert result == [0, 1, 2]

    def test_enum_mask_all_bits(self):
        result = list(enum_mask(15))  # 0b1111
        assert result == [0, 1, 2, 3]

    def test_enum_mask_with_mask_object(self):
        result = list(enum_mask(Mask(5)))
        assert result == [0, 2]


class TestGlobalToRelativeRank:
    def test_valid_conversions(self):
        # mask = 0b1011 (parties 0, 1, 3)
        mask = 11
        assert global_to_relative_rank(0, mask) == 0
        assert global_to_relative_rank(1, mask) == 1
        assert global_to_relative_rank(3, mask) == 2

    def test_single_bit_mask(self):
        mask = 4  # 0b100 (only party 2)
        assert global_to_relative_rank(2, mask) == 0

    def test_invalid_global_rank_not_in_mask(self):
        mask = 5  # 0b101 (parties 0, 2)
        with pytest.raises(ValueError):
            global_to_relative_rank(1, mask)

    def test_invalid_negative_global_rank(self):
        mask = 1
        with pytest.raises(ValueError):
            global_to_relative_rank(-1, mask)

    def test_complex_mask(self):
        mask = 0b10110101  # parties 0, 2, 4, 5, 7
        assert global_to_relative_rank(0, mask) == 0
        assert global_to_relative_rank(2, mask) == 1
        assert global_to_relative_rank(4, mask) == 2
        assert global_to_relative_rank(5, mask) == 3
        assert global_to_relative_rank(7, mask) == 4

    def test_with_mask_object(self):
        mask = Mask(11)  # 0b1011
        assert global_to_relative_rank(0, mask) == 0
        assert global_to_relative_rank(1, mask) == 1
        assert global_to_relative_rank(3, mask) == 2


class TestRelativeToGlobalRank:
    def test_valid_conversions(self):
        # mask = 0b1011 (parties 0, 1, 3)
        mask = 11
        assert relative_to_global_rank(0, mask) == 0
        assert relative_to_global_rank(1, mask) == 1
        assert relative_to_global_rank(2, mask) == 3

    def test_single_bit_mask(self):
        mask = 4  # 0b100 (only party 2)
        assert relative_to_global_rank(0, mask) == 2

    def test_invalid_relative_rank_too_large(self):
        mask = 5  # 0b101 (2 bits set)
        with pytest.raises(ValueError):
            relative_to_global_rank(2, mask)

    def test_invalid_negative_relative_rank(self):
        mask = 5
        with pytest.raises(ValueError):
            relative_to_global_rank(-1, mask)

    def test_complex_mask(self):
        mask = 0b10110101  # parties 0, 2, 4, 5, 7
        assert relative_to_global_rank(0, mask) == 0
        assert relative_to_global_rank(1, mask) == 2
        assert relative_to_global_rank(2, mask) == 4
        assert relative_to_global_rank(3, mask) == 5
        assert relative_to_global_rank(4, mask) == 7

    def test_with_mask_object(self):
        mask = Mask(11)  # 0b1011
        assert relative_to_global_rank(0, mask) == 0
        assert relative_to_global_rank(1, mask) == 1
        assert relative_to_global_rank(2, mask) == 3


class TestRoundTripConversion:
    @pytest.mark.parametrize("mask", [1, 3, 5, 7, 11, 15, 0b10110101])
    def test_global_to_relative_to_global(self, mask):
        for global_rank in enum_mask(mask):
            relative_rank = global_to_relative_rank(global_rank, mask)
            recovered_global = relative_to_global_rank(relative_rank, mask)
            assert recovered_global == global_rank

    @pytest.mark.parametrize("mask", [1, 3, 5, 7, 11, 15, 0b10110101])
    def test_relative_to_global_to_relative(self, mask):
        bit_count_val = bit_count(mask)
        for relative_rank in range(bit_count_val):
            global_rank = relative_to_global_rank(relative_rank, mask)
            recovered_relative = global_to_relative_rank(global_rank, mask)
            assert recovered_relative == relative_rank


class TestIsRankIn:
    def test_rank_in_mask(self):
        mask = 5  # 0b101
        assert is_rank_in(0, mask) is True
        assert is_rank_in(2, mask) is True

    def test_rank_not_in_mask(self):
        mask = 5  # 0b101
        assert is_rank_in(1, mask) is False
        assert is_rank_in(3, mask) is False

    def test_zero_mask(self):
        mask = 0
        assert is_rank_in(0, mask) is False
        assert is_rank_in(1, mask) is False

    def test_large_rank(self):
        mask = 1 << 10  # bit 10 set
        assert is_rank_in(10, mask) is True
        assert is_rank_in(9, mask) is False

    def test_with_mask_object(self):
        mask = Mask(5)  # 0b101
        assert is_rank_in(0, mask) is True
        assert is_rank_in(1, mask) is False


class TestEnsureRankIn:
    def test_rank_in_mask_no_exception(self):
        mask = 5  # 0b101
        ensure_rank_in(0, mask)  # Should not raise
        ensure_rank_in(2, mask)  # Should not raise

    def test_rank_not_in_mask_raises(self):
        mask = 5  # 0b101
        with pytest.raises(ValueError, match="Rank 1 is not in the party mask 5"):
            ensure_rank_in(1, mask)

    def test_with_mask_object(self):
        mask = Mask(5)  # 0b101
        ensure_rank_in(0, mask)  # Should not raise
        with pytest.raises(ValueError):
            ensure_rank_in(1, mask)


class TestIsSubset:
    def test_proper_subset(self):
        subset = 5  # 0b101
        superset = 7  # 0b111
        assert is_subset(subset, superset) is True

    def test_equal_sets(self):
        mask = 5  # 0b101
        assert is_subset(mask, mask) is True

    def test_not_subset(self):
        subset = 6  # 0b110
        superset = 5  # 0b101
        assert is_subset(subset, superset) is False

    def test_empty_subset(self):
        superset = 7  # 0b111
        assert is_subset(0, superset) is True

    def test_empty_superset(self):
        subset = 1
        assert is_subset(subset, 0) is False

    def test_disjoint_sets(self):
        subset = 5  # 0b101
        superset = 2  # 0b010
        assert is_subset(subset, superset) is False

    def test_with_mask_objects(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        assert is_subset(subset, superset) is True


class TestEnsureSubset:
    def test_valid_subset_no_exception(self):
        subset = 5  # 0b101
        superset = 7  # 0b111
        ensure_subset(subset, superset)  # Should not raise

    def test_equal_sets_no_exception(self):
        mask = 5  # 0b101
        ensure_subset(mask, mask)  # Should not raise

    def test_invalid_subset_raises(self):
        subset = 6  # 0b110
        superset = 5  # 0b101
        with pytest.raises(
            ValueError, match="Expect subset mask 6 to be a subset of superset mask 5"
        ):
            ensure_subset(subset, superset)

    def test_empty_subset_no_exception(self):
        superset = 7  # 0b111
        ensure_subset(0, superset)  # Should not raise

    def test_with_mask_objects(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        ensure_subset(subset, superset)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])
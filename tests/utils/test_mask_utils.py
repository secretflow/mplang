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

from mplang.utils.mask_utils import (
    bit_count,
    ensure_rank_in,
    ensure_subset,
    enum_mask,
    global_to_relative_rank,
    is_rank_in,
    is_subset,
    relative_to_global_rank,
)


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

    def test_bit_count_negative_mask(self):
        with pytest.raises(AssertionError):
            bit_count(-1)


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


class TestEnsureRankIn:
    def test_rank_in_mask_no_exception(self):
        mask = 5  # 0b101
        ensure_rank_in(0, mask)  # Should not raise
        ensure_rank_in(2, mask)  # Should not raise

    def test_rank_not_in_mask_raises(self):
        mask = 5  # 0b101
        with pytest.raises(ValueError, match="Rank 1 is not in the party mask 5"):
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


if __name__ == "__main__":
    pytest.main([__file__])

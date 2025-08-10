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

from mplang.core.base import Mask


class TestBitCount:
    def test_bit_count_zero(self):
        assert Mask(0).bit_count() == 0

    def test_bit_count_single_bit(self):
        assert Mask(1).bit_count() == 1
        assert Mask(2).bit_count() == 1
        assert Mask(4).bit_count() == 1
        assert Mask(8).bit_count() == 1

    def test_bit_count_multiple_bits(self):
        assert Mask(3).bit_count() == 2  # 0b11
        assert Mask(7).bit_count() == 3  # 0b111
        assert Mask(15).bit_count() == 4  # 0b1111
        assert Mask(5).bit_count() == 2  # 0b101

    def test_bit_count_large_numbers(self):
        assert Mask(255).bit_count() == 8  # 0b11111111
        assert Mask(1023).bit_count() == 10  # 0b1111111111

    def test_bit_count_negative_mask(self):
        with pytest.raises(ValueError):
            Mask(-1)


class TestEnumMask:
    def test_enum_mask_zero(self):
        result = list(Mask(0).enum())
        assert result == []

    def test_enum_mask_single_bit(self):
        result = list(Mask(1).enum())  # 0b1
        assert result == [0]

        result = list(Mask(4).enum())  # 0b100
        assert result == [2]

    def test_enum_mask_multiple_bits(self):
        result = list(Mask(5).enum())  # 0b101
        assert result == [0, 2]

        result = list(Mask(7).enum())  # 0b111
        assert result == [0, 1, 2]

    def test_enum_mask_all_bits(self):
        result = list(Mask(15).enum())  # 0b1111
        assert result == [0, 1, 2, 3]


class TestGlobalToRelativeRank:
    def test_valid_conversions(self):
        # mask = 0b1011 (parties 0, 1, 3)
        mask = Mask(11)
        assert mask.global_to_relative_rank(0) == 0
        assert mask.global_to_relative_rank(1) == 1
        assert mask.global_to_relative_rank(3) == 2

    def test_single_bit_mask(self):
        mask = Mask(4)  # 0b100 (only party 2)
        assert mask.global_to_relative_rank(2) == 0

    def test_invalid_global_rank_not_in_mask(self):
        mask = Mask(5)  # 0b101 (parties 0, 2)
        with pytest.raises(ValueError):
            mask.global_to_relative_rank(1)

    def test_invalid_negative_global_rank(self):
        mask = Mask(1)
        with pytest.raises(ValueError):
            mask.global_to_relative_rank(-1)

    def test_complex_mask(self):
        mask = Mask(0b10110101)  # parties 0, 2, 4, 5, 7
        assert mask.global_to_relative_rank(0) == 0
        assert mask.global_to_relative_rank(2) == 1
        assert mask.global_to_relative_rank(4) == 2
        assert mask.global_to_relative_rank(5) == 3
        assert mask.global_to_relative_rank(7) == 4


class TestRelativeToGlobalRank:
    def test_valid_conversions(self):
        # mask = 0b1011 (parties 0, 1, 3)
        mask = Mask(11)
        assert mask.relative_to_global_rank(0) == 0
        assert mask.relative_to_global_rank(1) == 1
        assert mask.relative_to_global_rank(2) == 3

    def test_single_bit_mask(self):
        mask = Mask(4)  # 0b100 (only party 2)
        assert mask.relative_to_global_rank(0) == 2

    def test_invalid_relative_rank_too_large(self):
        mask = Mask(5)  # 0b101 (2 bits set)
        with pytest.raises(ValueError):
            mask.relative_to_global_rank(2)

    def test_invalid_negative_relative_rank(self):
        mask = Mask(5)
        with pytest.raises(ValueError):
            mask.relative_to_global_rank(-1)

    def test_complex_mask(self):
        mask = Mask(0b10110101)  # parties 0, 2, 4, 5, 7
        assert mask.relative_to_global_rank(0) == 0
        assert mask.relative_to_global_rank(1) == 2
        assert mask.relative_to_global_rank(2) == 4
        assert mask.relative_to_global_rank(3) == 5
        assert mask.relative_to_global_rank(4) == 7


class TestRoundTripConversion:
    @pytest.mark.parametrize("mask", [1, 3, 5, 7, 11, 15, 0b10110101])
    def test_global_to_relative_to_global(self, mask):
        mask_obj = Mask(mask)
        for global_rank in mask_obj.enum():
            relative_rank = mask_obj.global_to_relative_rank(global_rank)
            recovered_global = mask_obj.relative_to_global_rank(relative_rank)
            assert recovered_global == global_rank

    @pytest.mark.parametrize("mask", [1, 3, 5, 7, 11, 15, 0b10110101])
    def test_relative_to_global_to_relative(self, mask):
        mask_obj = Mask(mask)
        bit_count_val = mask_obj.bit_count()
        for relative_rank in range(bit_count_val):
            global_rank = mask_obj.relative_to_global_rank(relative_rank)
            recovered_relative = mask_obj.global_to_relative_rank(global_rank)
            assert recovered_relative == relative_rank


class TestIsRankIn:
    def test_rank_in_mask(self):
        mask = Mask(5)  # 0b101
        assert mask.contains_rank(0) is True
        assert mask.contains_rank(2) is True

    def test_rank_not_in_mask(self):
        mask = Mask(5)  # 0b101
        assert mask.contains_rank(1) is False
        assert mask.contains_rank(3) is False

    def test_zero_mask(self):
        mask = Mask(0)
        assert mask.contains_rank(0) is False
        assert mask.contains_rank(1) is False

    def test_large_rank(self):
        mask = Mask(1 << 10)  # bit 10 set
        assert mask.contains_rank(10) is True
        assert mask.contains_rank(9) is False


class TestEnsureRankIn:
    def test_rank_in_mask_no_exception(self):
        mask = Mask(5)  # 0b101
        mask.ensure_rank_in(0)  # Should not raise
        mask.ensure_rank_in(2)  # Should not raise

    def test_rank_not_in_mask_raises(self):
        mask = Mask(5)  # 0b101
        with pytest.raises(ValueError, match="Rank 1 is not in the party mask 5"):
            mask.ensure_rank_in(1)


class TestIsSubset:
    def test_proper_subset(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        assert subset.is_subset_of(superset) is True

    def test_equal_sets(self):
        mask = Mask(5)  # 0b101
        assert mask.is_subset_of(mask) is True

    def test_not_subset(self):
        subset = Mask(6)  # 0b110
        superset = Mask(5)  # 0b101
        assert subset.is_subset_of(superset) is False

    def test_empty_subset(self):
        superset = Mask(7)  # 0b111
        assert Mask(0).is_subset_of(superset) is True

    def test_empty_superset(self):
        subset = Mask(1)
        assert subset.is_subset_of(Mask(0)) is False

    def test_disjoint_sets(self):
        subset = Mask(5)  # 0b101
        superset = Mask(2)  # 0b010
        assert subset.is_subset_of(superset) is False


class TestEnsureSubset:
    def test_valid_subset_no_exception(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        subset.ensure_subset_of(superset)  # Should not raise

    def test_equal_sets_no_exception(self):
        mask = Mask(5)  # 0b101
        mask.ensure_subset_of(mask)  # Should not raise

    def test_invalid_subset_raises(self):
        subset = Mask(6)  # 0b110
        superset = Mask(5)  # 0b101
        with pytest.raises(
            ValueError, match="Expect subset mask 6 to be a subset of superset mask 5"
        ):
            subset.ensure_subset_of(superset)

    def test_empty_subset_no_exception(self):
        superset = Mask(7)  # 0b111
        Mask(0).ensure_subset_of(superset)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])

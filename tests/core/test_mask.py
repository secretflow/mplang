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

from mplang.core.mask import Mask


class TestMask:
    """Test cases for the new Mask class."""

    def test_constructor(self):
        mask = Mask(5)
        assert mask.value == 5

    def test_constructor_negative_value(self):
        with pytest.raises(ValueError, match="Mask value must be non-negative"):
            Mask(-1)

    def test_from_int(self):
        mask = Mask.from_int(3)
        assert mask.value == 3

    def test_from_single_rank(self):
        mask = Mask.from_ranks(2)
        assert mask.value == 4  # 1 << 2

    def test_from_ranks(self):
        mask = Mask.from_ranks([0, 2, 3])
        assert mask.value == 13  # 1 | 4 | 8

    def test_from_ranks_negative(self):
        with pytest.raises(ValueError, match="All ranks must be non-negative"):
            Mask.from_ranks([0, -1, 2])

    def test_all(self):
        mask = Mask.all(3)
        assert mask.value == 7  # 0b111

    def test_all_zero(self):
        mask = Mask.all(0)
        assert mask.value == 0

    def test_none(self):
        mask = Mask.none()
        assert mask.value == 0

    def test_num_parties(self):
        assert Mask(0).num_parties() == 0
        assert Mask(1).num_parties() == 1
        assert Mask(3).num_parties() == 2  # 0b11
        assert Mask(5).num_parties() == 2  # 0b101
        assert Mask(7).num_parties() == 3  # 0b111

    def test_ranks_iteration(self):
        assert list(Mask(0)) == []
        assert list(Mask(1)) == [0]
        assert list(Mask(4)) == [2]
        assert list(Mask(5)) == [0, 2]  # 0b101
        assert list(Mask(7)) == [0, 1, 2]  # 0b111

    def test_contains(self):
        mask = Mask(5)  # 0b101
        assert 0 in mask
        assert 2 in mask
        assert 1 not in mask
        assert 3 not in mask
        assert -1 not in mask

    def test_is_disjoint(self):
        mask1 = Mask(5)  # 0b101
        mask2 = Mask(2)  # 0b010
        assert mask1.is_disjoint(mask2)
        assert mask2.is_disjoint(mask1)
        assert not mask1.is_disjoint(Mask(1))  # 0b001

    def test_is_subset(self):
        mask1 = Mask(5)  # 0b101
        mask2 = Mask(7)  # 0b111
        assert mask1.is_subset(mask2)
        assert not mask2.is_subset(mask1)
        assert mask1.is_subset(mask1)

    def test_is_superset(self):
        mask1 = Mask(5)  # 0b101
        mask2 = Mask(7)  # 0b111
        assert mask2.is_superset(mask1)
        assert not mask1.is_superset(mask2)
        assert mask1.is_superset(mask1)

    def test_union(self):
        mask1 = Mask(5)  # 0b101
        mask2 = Mask(2)  # 0b010
        result = mask1.union(mask2)
        assert result.value == 7  # 0b111

    def test_intersection(self):
        mask1 = Mask(5)  # 0b101
        mask2 = Mask(7)  # 0b111
        result = mask1.intersection(mask2)
        assert result.value == 5  # 0b101

    def test_difference(self):
        mask1 = Mask(7)  # 0b111
        mask2 = Mask(5)  # 0b101
        result = mask1.difference(mask2)
        assert result.value == 2  # 0b010

    def test_xor(self):
        mask1 = Mask(5)  # 0b101
        mask2 = Mask(3)  # 0b011
        result = mask1 ^ mask2
        assert result.value == 6  # 0b110

    def test_invert(self):
        mask = Mask(5)  # 0b101
        result = ~mask
        # After masking to 64 bits: ~5 & ((1 << 64) - 1)
        expected = ~5 & ((1 << 64) - 1)
        assert result.value == expected

    def test_global_to_relative_rank(self):
        mask = Mask(0b10110101)  # parties 0, 2, 4, 5, 7
        assert mask.global_to_relative_rank(0) == 0
        assert mask.global_to_relative_rank(2) == 1
        assert mask.global_to_relative_rank(4) == 2
        assert mask.global_to_relative_rank(5) == 3
        assert mask.global_to_relative_rank(7) == 4

    def test_relative_to_global_rank(self):
        mask = Mask(11)  # 0b1011
        assert mask.relative_to_global_rank(0) == 0
        assert mask.relative_to_global_rank(1) == 1
        assert mask.relative_to_global_rank(2) == 3

    def test_round_trip_conversion(self):
        mask = Mask(0b10110101)
        for global_rank in mask:
            relative_rank = mask.global_to_relative_rank(global_rank)
            recovered_global = mask.relative_to_global_rank(relative_rank)
            assert recovered_global == global_rank

    def test_equality(self):
        assert Mask(5) == Mask(5)
        assert Mask(5) == 5
        assert Mask(5) != Mask(3)
        assert Mask(5) != 3

    def test_int_conversion(self):
        mask = Mask(5)
        assert int(mask) == 5

    def test_str_representation(self):
        mask = Mask(5)
        assert str(mask) == "Mask([0, 2])"
        assert repr(mask) == "Mask(0b101)"

    def test_is_empty(self):
        assert Mask(0).is_empty
        assert not Mask(1).is_empty

    def test_is_single(self):
        assert Mask(1).is_single
        assert Mask(2).is_single
        assert not Mask(3).is_single  # 0b11
        assert not Mask(0).is_single

    def test_json_serialization(self):
        mask = Mask(5)
        serialized = mask.to_json()
        deserialized = Mask.from_json(serialized)
        assert mask == deserialized

    def test_copy(self):
        mask = Mask(5)
        copy = mask.copy()
        assert mask == copy
        assert mask is not copy

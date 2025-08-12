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

try:
    from mplang.core.base import Mask
except ImportError as e:
    # If mplang dependencies are not available, skip these tests
    pytest.skip(f"Skipping mask tests due to missing dependencies: {e}", allow_module_level=True)


class TestMaskClass:
    def test_mask_creation(self):
        m1 = Mask(5)
        assert m1.value == 5
        assert int(m1) == 5
        
        m2 = Mask(m1)  # Copy constructor
        assert m2.value == 5
        
        # Test validation
        with pytest.raises(ValueError):
            Mask(-1)
    
    def test_bit_count(self):
        assert Mask(0).bit_count() == 0
        assert Mask(1).bit_count() == 1
        assert Mask(3).bit_count() == 2  # 0b11
        assert Mask(7).bit_count() == 3  # 0b111
        assert Mask(5).bit_count() == 2  # 0b101
        assert Mask(255).bit_count() == 8  # 0b11111111

    def test_enum_mask(self):
        result = list(Mask(0).enum_mask())
        assert result == []

        result = list(Mask(1).enum_mask())  # 0b1
        assert result == [0]

        result = list(Mask(4).enum_mask())  # 0b100
        assert result == [2]

        result = list(Mask(5).enum_mask())  # 0b101
        assert result == [0, 2]

        result = list(Mask(7).enum_mask())  # 0b111
        assert result == [0, 1, 2]

    def test_is_rank_in(self):
        mask = Mask(5)  # 0b101
        assert mask.is_rank_in(0) is True
        assert mask.is_rank_in(1) is False
        assert mask.is_rank_in(2) is True
        assert mask.is_rank_in(3) is False

        mask = Mask(0)
        assert mask.is_rank_in(0) is False
        assert mask.is_rank_in(1) is False

    def test_ensure_rank_in(self):
        mask = Mask(5)  # 0b101
        mask.ensure_rank_in(0)  # Should not raise
        mask.ensure_rank_in(2)  # Should not raise

        with pytest.raises(ValueError, match="Rank 1 is not in the party mask 5"):
            mask.ensure_rank_in(1)

    def test_is_subset(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        assert subset.is_subset(superset) is True
        
        # Test with int
        assert subset.is_subset(7) is True

        # Equal sets
        mask = Mask(5)
        assert mask.is_subset(mask) is True

        # Not subset
        subset = Mask(6)  # 0b110
        superset = Mask(5)  # 0b101
        assert subset.is_subset(superset) is False

        # Empty subset
        assert Mask(0).is_subset(Mask(7)) is True

    def test_ensure_subset(self):
        subset = Mask(5)  # 0b101
        superset = Mask(7)  # 0b111
        subset.ensure_subset(superset)  # Should not raise

        subset = Mask(6)  # 0b110
        superset = Mask(5)  # 0b101
        with pytest.raises(ValueError, match="Expect subset mask 6 to be a subset of superset mask 5"):
            subset.ensure_subset(superset)

    def test_global_to_relative_rank(self):
        # mask = 0b1011 (parties 0, 1, 3)
        mask = Mask(11)
        assert mask.global_to_relative_rank(0) == 0
        assert mask.global_to_relative_rank(1) == 1
        assert mask.global_to_relative_rank(3) == 2

        mask = Mask(4)  # 0b100 (only party 2)
        assert mask.global_to_relative_rank(2) == 0

        with pytest.raises(ValueError):
            mask.global_to_relative_rank(1)  # Not in mask

    def test_relative_to_global_rank(self):
        # mask = 0b1011 (parties 0, 1, 3)
        mask = Mask(11)
        assert mask.relative_to_global_rank(0) == 0
        assert mask.relative_to_global_rank(1) == 1
        assert mask.relative_to_global_rank(2) == 3

        mask = Mask(4)  # 0b100 (only party 2)
        assert mask.relative_to_global_rank(0) == 2

        with pytest.raises(ValueError):
            mask.relative_to_global_rank(2)  # Index too large

    def test_round_trip_conversion(self):
        mask = Mask(0b10110101)  # parties 0, 2, 4, 5, 7
        
        for global_rank in mask.enum_mask():
            relative_rank = mask.global_to_relative_rank(global_rank)
            recovered_global = mask.relative_to_global_rank(relative_rank)
            assert recovered_global == global_rank

    def test_bitwise_operations(self):
        m1 = Mask(5)  # 0b101
        m2 = Mask(3)  # 0b011

        # OR operation (union)
        result = m1 | m2
        assert result.value == 7  # 0b111
        assert isinstance(result, Mask)

        # AND operation (intersection)
        result = m1 & m2
        assert result.value == 1  # 0b001
        assert isinstance(result, Mask)

        # XOR operation
        result = m1 ^ m2
        assert result.value == 6  # 0b110
        assert isinstance(result, Mask)

        # Test with int
        result = m1 | 2
        assert result.value == 7  # 0b111

    def test_comparison_operations(self):
        m1 = Mask(5)
        m2 = Mask(5)
        m3 = Mask(3)

        assert m1 == m2
        assert m1 != m3
        assert m1 == 5  # Test with int
        assert m1 != 3

    def test_class_methods(self):
        # Test union
        result = Mask.union(Mask(5), Mask(3))
        assert result.value == 7

        result = Mask.union(5, 3)  # Test with ints
        assert result.value == 7

        # Test intersection
        result = Mask.intersection(Mask(5), Mask(3))
        assert result.value == 1

        result = Mask.intersection()  # Empty
        assert result.value == 0

        # Test is_disjoint
        assert Mask.is_disjoint(Mask(1), Mask(2), Mask(4)) is True
        assert Mask.is_disjoint(Mask(1), Mask(3)) is False  # 1 & 3 = 1

    def test_string_representation(self):
        mask = Mask(5)
        assert str(mask) == "5"
        assert "Mask(5=0b101)" in repr(mask)

    def test_boolean_conversion(self):
        assert bool(Mask(0)) is False
        assert bool(Mask(1)) is True
        assert bool(Mask(5)) is True


if __name__ == "__main__":
    pytest.main([__file__])
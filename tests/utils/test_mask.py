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

import os

# Direct import to avoid mplang init dependencies
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mplang', 'utils'))

from mask import Mask


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

    def test_mask_from_ranks(self):
        m = Mask.from_ranks(0, 2, 4)
        assert m.value == 21  # 0b10101
        assert list(m.enum()) == [0, 2, 4]

    def test_mask_from_ranks_empty(self):
        m = Mask.from_ranks()
        assert m.value == 0
        assert m.is_empty() is True

    def test_mask_from_ranks_invalid(self):
        with pytest.raises(ValueError):
            Mask.from_ranks(-1)

    def test_mask_full(self):
        m = Mask.full(3)
        assert m.value == 7  # 0b111
        assert m.size() == 3

    def test_mask_full_invalid(self):
        with pytest.raises(ValueError):
            Mask.full(0)
        with pytest.raises(ValueError):
            Mask.full(-1)

    def test_mask_empty(self):
        m = Mask.empty()
        assert m.value == 0
        assert m.is_empty() is True

    def test_mask_to_ranks(self):
        m = Mask(5)  # 0b101
        assert m.to_ranks() == [0, 2]

    def test_mask_size(self):
        assert Mask(5).size() == 2
        assert Mask(7).size() == 3
        assert Mask(0).size() == 0

    def test_mask_is_empty(self):
        assert Mask(0).is_empty() is True
        assert Mask(1).is_empty() is False

    def test_mask_is_single(self):
        assert Mask(1).is_single() is True  # 0b1
        assert Mask(4).is_single() is True  # 0b100
        assert Mask(5).is_single() is False  # 0b101
        assert Mask(0).is_single() is False

    def test_mask_copy(self):
        m1 = Mask(5)
        m2 = m1.copy()
        assert m1 == m2
        assert m1 is not m2


if __name__ == "__main__":
    pytest.main([__file__])

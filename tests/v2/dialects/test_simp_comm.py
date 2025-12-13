# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SIMP communication primitives (shuffle_dynamic, shuffle, converge)."""

import numpy as np
import pytest

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects.simp import converge, shuffle_dynamic, shuffle_static
from mplang.v2.runtime.interpreter import InterpObject

pytestmark = pytest.mark.usefixtures("simp_simulator_default")


class TestPshfl:
    """Tests for dynamic shuffle primitive."""

    def test_pshfl_creates_dynamic_mask(self):
        """Test that shuffle_dynamic output has dynamic mask (parties=None)."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)])
        index = InterpObject(np.array(0), elt.MPType[elt.Tensor[elt.i32, ()], (0, 1)])

        traced = el.trace(lambda s, i: shuffle_dynamic(s, i), src, index)

        assert len(traced.graph.outputs) == 1
        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties is None  # Dynamic mask!
        assert output_type.value_type == elt.Tensor[elt.f32, ()]

    def test_pshfl_requires_mp_typed_src(self):
        """Test that pshfl raises TypeError for non-MP src."""
        src = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
        index = InterpObject(np.array(0), elt.MPType[elt.Tensor[elt.i32, ()], (0,)])

        with pytest.raises(TypeError, match="shuffle_dynamic requires MP-typed src"):
            el.trace(lambda s, i: shuffle_dynamic(s, i), src, index)

    def test_pshfl_requires_mp_typed_index(self):
        """Test that pshfl raises TypeError for non-MP index."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0,)])
        index = InterpObject(np.array(0), elt.Tensor[elt.i32, ()])

        with pytest.raises(TypeError, match="shuffle_dynamic requires MP-typed index"):
            el.trace(lambda s, i: shuffle_dynamic(s, i), src, index)

    def test_pshfl_requires_scalar_index(self):
        """Test that pshfl raises TypeError for non-scalar index."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0,)])
        index = InterpObject(
            np.array([0, 1]), elt.MPType[elt.Tensor[elt.i32, (2,)], (0,)]
        )

        with pytest.raises(TypeError, match="shuffle_dynamic index must be scalar"):
            el.trace(lambda s, i: shuffle_dynamic(s, i), src, index)


class TestPshflS:
    """Tests for static shuffle primitive."""

    def test_pshfl_s_creates_static_mask(self):
        """Test that shuffle_static output has static mask."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)])

        traced = el.trace(lambda s: shuffle_static(s, routing={0: 1}), src)

        assert len(traced.graph.outputs) == 1
        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties == (0,)  # Static mask!
        assert output_type.value_type == elt.Tensor[elt.f32, ()]

    def test_pshfl_s_requires_mp_typed_src(self):
        """Test that shuffle_static raises TypeError for non-MP src."""
        src = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])

        with pytest.raises(TypeError, match="shuffle_static requires MP-typed src"):
            el.trace(lambda s: shuffle_static(s, routing={0: 1}), src)

    def test_pshfl_s_requires_nonempty_routing(self):
        """Test that shuffle_static raises ValueError for empty routing."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)])

        with pytest.raises(
            ValueError, match="shuffle_static requires non-empty routing dict"
        ):
            el.trace(lambda s: shuffle_static(s, routing={}), src)

    def test_pshfl_s_requires_dict_routing(self):
        """Test that shuffle_static raises TypeError for non-dict routing."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)])

        with pytest.raises(TypeError, match="shuffle_static requires routing dict"):
            el.trace(lambda s: shuffle_static(s, routing=[1]), src)

    def test_pshfl_s_validates_src_ranks_in_src_parties(self):
        """Test that shuffle_static raises ValueError when source not in src.parties."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)])

        with pytest.raises(ValueError, match=r"routing\[0\]=2 not in src\.parties"):
            el.trace(lambda s: shuffle_static(s, routing={0: 2}), src)

    def test_pshfl_s_allows_src_ranks_when_src_parties_none(self):
        """Test that shuffle_static allows any source when src.parties is None."""
        src = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], None])

        # Should not raise even though source=2 is not in src.parties (None)
        traced = el.trace(lambda s: shuffle_static(s, routing={0: 2}), src)
        assert traced.graph.outputs[0].type.parties == (0,)


class TestPconv:
    """Tests for converge primitive."""

    def test_pconv_unions_static_parties(self):
        """Test that pconv unions disjoint static parties."""
        x = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0,)])
        y = InterpObject(np.array(2.0), elt.MPType[elt.Tensor[elt.f32, ()], (1,)])

        traced = el.trace(lambda a, b: converge(a, b), x, y)

        assert len(traced.graph.outputs) == 1
        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties == (0, 1)  # Union of (0,) and (1,)
        assert output_type.value_type == elt.Tensor[elt.f32, ()]

    def test_pconv_propagates_dynamic_mask(self):
        """Test that pconv propagates dynamic mask when any input has None parties."""
        x = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0,)])
        y = InterpObject(np.array(2.0), elt.MPType[elt.Tensor[elt.f32, ()], None])

        traced = el.trace(lambda a, b: converge(a, b), x, y)

        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties is None  # Dynamic!

    def test_pconv_rejects_overlapping_parties(self):
        """Test that pconv raises ValueError for overlapping parties."""
        x = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0, 1)])
        y = InterpObject(np.array(2.0), elt.MPType[elt.Tensor[elt.f32, ()], (1, 2)])

        with pytest.raises(ValueError, match="converge requires disjoint parties"):
            el.trace(lambda a, b: converge(a, b), x, y)

    def test_pconv_requires_mp_typed_inputs(self):
        """Test that pconv raises TypeError for non-MP inputs."""
        x = InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
        y = InterpObject(np.array(2.0), elt.MPType[elt.Tensor[elt.f32, ()], (1,)])

        with pytest.raises(TypeError, match="converge input 0 must be MP-typed"):
            el.trace(lambda a, b: converge(a, b), x, y)

    def test_pconv_requires_consistent_value_types(self):
        """Test that pconv raises TypeError for inconsistent value types."""
        x = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0,)])
        y = InterpObject(np.array(2), elt.MPType[elt.Tensor[elt.i32, ()], (1,)])

        with pytest.raises(TypeError, match="converge value type mismatch"):
            el.trace(lambda a, b: converge(a, b), x, y)

    def test_pconv_requires_at_least_one_input(self):
        """Test that pconv raises TypeError when called with no arguments."""
        with pytest.raises(TypeError, match="converge requires at least one input"):
            el.trace(lambda: converge())

    def test_pconv_handles_three_disjoint_inputs(self):
        """Test that pconv correctly unions three disjoint parties."""
        x = InterpObject(np.array(1.0), elt.MPType[elt.Tensor[elt.f32, ()], (0,)])
        y = InterpObject(np.array(2.0), elt.MPType[elt.Tensor[elt.f32, ()], (1,)])
        z = InterpObject(np.array(3.0), elt.MPType[elt.Tensor[elt.f32, ()], (2,)])

        traced = el.trace(lambda a, b, c: converge(a, b, c), x, y, z)

        output_type = traced.graph.outputs[0].type
        assert output_type.parties == (0, 1, 2)

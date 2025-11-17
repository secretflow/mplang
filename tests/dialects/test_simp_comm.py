"""Tests for SIMP communication primitives (pshfl, pshfl_s, pconv)."""

import numpy as np
import pytest

import mplang.edsl as el
import mplang.edsl.typing as elt
from mplang.dialects.simp import pconv, pshfl, pshfl_s


class TestPshfl:
    """Tests for dynamic shuffle primitive."""

    def test_pshfl_creates_dynamic_mask(self):
        """Test that pshfl output has dynamic mask (parties=None)."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0, 1)])
        index = el.InterpObject(np.array(0), elt.MP[elt.Tensor[elt.i32, ()], (0, 1)])

        traced = el.trace(lambda s, i: pshfl(s, i), src, index)

        assert len(traced.graph.outputs) == 1
        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties is None  # Dynamic mask!
        assert output_type.value_type == elt.Tensor[elt.f32, ()]

    def test_pshfl_requires_mp_typed_src(self):
        """Test that pshfl raises TypeError for non-MP src."""
        src = el.InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
        index = el.InterpObject(np.array(0), elt.MP[elt.Tensor[elt.i32, ()], (0,)])

        with pytest.raises(TypeError, match="shuffle_dynamic requires MP-typed src"):
            el.trace(lambda s, i: pshfl(s, i), src, index)

    def test_pshfl_requires_mp_typed_index(self):
        """Test that pshfl raises TypeError for non-MP index."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0,)])
        index = el.InterpObject(np.array(0), elt.Tensor[elt.i32, ()])

        with pytest.raises(TypeError, match="shuffle_dynamic requires MP-typed index"):
            el.trace(lambda s, i: pshfl(s, i), src, index)

    def test_pshfl_requires_scalar_index(self):
        """Test that pshfl raises TypeError for non-scalar index."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0,)])
        index = el.InterpObject(
            np.array([0, 1]), elt.MP[elt.Tensor[elt.i32, (2,)], (0,)]
        )

        with pytest.raises(TypeError, match="shuffle_dynamic index must be scalar"):
            el.trace(lambda s, i: pshfl(s, i), src, index)


class TestPshflS:
    """Tests for static shuffle primitive."""

    def test_pshfl_s_creates_static_mask(self):
        """Test that pshfl_s output has static mask."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0, 1)])

        traced = el.trace(lambda s: pshfl_s(s, parties=(0,), src_ranks=[1]), src)

        assert len(traced.graph.outputs) == 1
        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties == (0,)  # Static mask!
        assert output_type.value_type == elt.Tensor[elt.f32, ()]

    def test_pshfl_s_requires_mp_typed_src(self):
        """Test that pshfl_s raises TypeError for non-MP src."""
        src = el.InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])

        with pytest.raises(TypeError, match="shuffle requires MP-typed src"):
            el.trace(lambda s: pshfl_s(s, parties=(0,), src_ranks=[1]), src)

    def test_pshfl_s_requires_explicit_parties(self):
        """Test that pshfl_s raises TypeError when parties=None."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0, 1)])

        with pytest.raises(TypeError, match="shuffle requires explicit parties"):
            el.trace(lambda s: pshfl_s(s, parties=None, src_ranks=[1]), src)

    def test_pshfl_s_validates_src_ranks_length(self):
        """Test that pshfl_s raises ValueError when src_ranks length mismatches."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0, 1)])

        with pytest.raises(ValueError, match=r"src_ranks length .* != parties count"):
            el.trace(lambda s: pshfl_s(s, parties=(0, 1), src_ranks=[1]), src)

    def test_pshfl_s_validates_src_ranks_in_src_parties(self):
        """Test that pshfl_s raises ValueError when src_rank not in src.parties."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0, 1)])

        with pytest.raises(ValueError, match=r"src_rank 2 not in src\.parties"):
            el.trace(lambda s: pshfl_s(s, parties=(0,), src_ranks=[2]), src)

    def test_pshfl_s_allows_src_ranks_when_src_parties_none(self):
        """Test that pshfl_s allows any src_ranks when src.parties is None."""
        src = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], None])

        # Should not raise even though src_rank=2 is not in src.parties (None)
        traced = el.trace(lambda s: pshfl_s(s, parties=(0,), src_ranks=[2]), src)
        assert traced.graph.outputs[0].type.parties == (0,)


class TestPconv:
    """Tests for converge primitive."""

    def test_pconv_unions_static_parties(self):
        """Test that pconv unions disjoint static parties."""
        x = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0,)])
        y = el.InterpObject(np.array(2.0), elt.MP[elt.Tensor[elt.f32, ()], (1,)])

        traced = el.trace(lambda a, b: pconv(a, b), x, y)

        assert len(traced.graph.outputs) == 1
        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties == (0, 1)  # Union of (0,) and (1,)
        assert output_type.value_type == elt.Tensor[elt.f32, ()]

    def test_pconv_propagates_dynamic_mask(self):
        """Test that pconv propagates dynamic mask when any input has None parties."""
        x = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0,)])
        y = el.InterpObject(np.array(2.0), elt.MP[elt.Tensor[elt.f32, ()], None])

        traced = el.trace(lambda a, b: pconv(a, b), x, y)

        output_type = traced.graph.outputs[0].type
        assert isinstance(output_type, elt.MPType)
        assert output_type.parties is None  # Dynamic!

    def test_pconv_rejects_overlapping_parties(self):
        """Test that pconv raises ValueError for overlapping parties."""
        x = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0, 1)])
        y = el.InterpObject(np.array(2.0), elt.MP[elt.Tensor[elt.f32, ()], (1, 2)])

        with pytest.raises(ValueError, match="converge requires disjoint parties"):
            el.trace(lambda a, b: pconv(a, b), x, y)

    def test_pconv_requires_mp_typed_inputs(self):
        """Test that pconv raises TypeError for non-MP inputs."""
        x = el.InterpObject(np.array(1.0), elt.Tensor[elt.f32, ()])
        y = el.InterpObject(np.array(2.0), elt.MP[elt.Tensor[elt.f32, ()], (1,)])

        with pytest.raises(TypeError, match="converge input 0 must be MP-typed"):
            el.trace(lambda a, b: pconv(a, b), x, y)

    def test_pconv_requires_consistent_value_types(self):
        """Test that pconv raises TypeError for inconsistent value types."""
        x = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0,)])
        y = el.InterpObject(np.array(2), elt.MP[elt.Tensor[elt.i32, ()], (1,)])

        with pytest.raises(TypeError, match="converge value type mismatch"):
            el.trace(lambda a, b: pconv(a, b), x, y)

    def test_pconv_requires_at_least_one_input(self):
        """Test that pconv raises TypeError when called with no arguments."""
        with pytest.raises(TypeError, match="converge requires at least one input"):
            el.trace(lambda: pconv())

    def test_pconv_handles_three_disjoint_inputs(self):
        """Test that pconv correctly unions three disjoint parties."""
        x = el.InterpObject(np.array(1.0), elt.MP[elt.Tensor[elt.f32, ()], (0,)])
        y = el.InterpObject(np.array(2.0), elt.MP[elt.Tensor[elt.f32, ()], (1,)])
        z = el.InterpObject(np.array(3.0), elt.MP[elt.Tensor[elt.f32, ()], (2,)])

        traced = el.trace(lambda a, b, c: pconv(a, b, c), x, y, z)

        output_type = traced.graph.outputs[0].type
        assert output_type.parties == (0, 1, 2)

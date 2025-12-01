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

"""Tests for collective communication library."""

import pytest

from mplang.v2.dialects.simp import constant, converge
from mplang.v2.edsl import Tracer
from mplang.v2.edsl.typing import MPType
from mplang.v2.libs.collective import (
    allreplicate,
    collect,
    distribute,
    permute,
    replicate,
    transfer,
)


class TestTransfer:
    """Tests for transfer (P2P) operation."""

    def test_transfer_basic(self) -> None:
        """Transfer data from party 0 to party 1."""
        tracer = Tracer()
        with tracer:
            x = constant((0,), 42)
            y = transfer(x, to=1)

        assert isinstance(y.type, MPType)
        assert y.type.parties == (1,)

    def test_transfer_same_party_returns_input(self) -> None:
        """Transfer to same party returns input unchanged."""
        tracer = Tracer()
        with tracer:
            x = constant((0,), 42)
            y = transfer(x, to=0)

        # Should return same object (no shuffle needed)
        assert y is x

    def test_transfer_requires_single_party(self) -> None:
        """Transfer fails if source has multiple parties."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x = converge(x0, x1)

            with pytest.raises(ValueError, match="single-party"):
                transfer(x, to=2)


class TestReplicate:
    """Tests for replicate (broadcast) operation."""

    def test_replicate_basic(self) -> None:
        """Replicate data from party 0 to multiple parties."""
        tracer = Tracer()
        with tracer:
            x = constant((0,), 42)
            y = replicate(x, to=(0, 1, 2))

        assert isinstance(y.type, MPType)
        assert y.type.parties == (0, 1, 2)

    def test_replicate_preserves_value_type(self) -> None:
        """Replicate preserves the underlying value type."""
        tracer = Tracer()
        with tracer:
            x = constant((0,), 42)
            y = replicate(x, to=(1, 2))

        assert isinstance(y.type, MPType)
        assert y.type.value_type == x.type.value_type

    def test_replicate_requires_single_party(self) -> None:
        """Replicate fails if source has multiple parties."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x = converge(x0, x1)

            with pytest.raises(ValueError, match="single-party"):
                replicate(x, to=(0, 1, 2))


class TestDistribute:
    """Tests for distribute (scatter) operation."""

    def test_distribute_basic(self) -> None:
        """Distribute values from party 0 to multiple parties."""
        tracer = Tracer()
        with tracer:
            values = [constant((0,), i) for i in range(3)]
            y = distribute(values, frm=0)

        assert isinstance(y.type, MPType)
        assert y.type.parties == (0, 1, 2)

    def test_distribute_empty_fails(self) -> None:
        """Distribute fails with empty list."""
        tracer = Tracer()
        with tracer:
            with pytest.raises(ValueError, match="at least one"):
                distribute([], frm=0)

    def test_distribute_wrong_source_fails(self) -> None:
        """Distribute fails if value is not from specified party."""
        tracer = Tracer()
        with tracer:
            values = [constant((1,), 42)]  # held by party 1, not 0

            with pytest.raises(ValueError, match="party 0"):
                distribute(values, frm=0)


class TestCollect:
    """Tests for collect (gather) operation."""

    def test_collect_basic(self) -> None:
        """Collect distributed data to one party."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x2 = constant((2,), 3)
            x = converge(x0, x1, x2)

            ys = collect(x, to=0)

        assert len(ys) == 3
        for y in ys:
            assert isinstance(y.type, MPType)
            assert y.type.parties == (0,)

    def test_collect_preserves_order(self) -> None:
        """Collect preserves source party order."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x = converge(x0, x1)

            ys = collect(x, to=1)

        # ys[0] came from party 0, ys[1] came from party 1
        assert len(ys) == 2
        # Both now held by party 1
        assert ys[0].type.parties == (1,)
        assert ys[1].type.parties == (1,)


class TestAllreplicate:
    """Tests for allreplicate (allgather-like) operation."""

    def test_allreplicate_basic(self) -> None:
        """Allreplicate replicates each party's data to all."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x = converge(x0, x1)

            ys = allreplicate(x)

        assert len(ys) == 2
        # Each result is replicated to all original parties
        for y in ys:
            assert isinstance(y.type, MPType)
            assert y.type.parties == (0, 1)


class TestPermute:
    """Tests for permute operation."""

    def test_permute_swap(self) -> None:
        """Permute can swap party data."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x = converge(x0, x1)

            y = permute(x, mapping={0: 1, 1: 0})

        assert isinstance(y.type, MPType)
        assert y.type.parties == (0, 1)

    def test_permute_subset(self) -> None:
        """Permute can select a subset of parties."""
        tracer = Tracer()
        with tracer:
            x0 = constant((0,), 1)
            x1 = constant((1,), 2)
            x2 = constant((2,), 3)
            x = converge(x0, x1, x2)

            # Only extract party 1's data to party 0
            y = permute(x, mapping={0: 1})

        assert isinstance(y.type, MPType)
        assert y.type.parties == (0,)


class TestGraphGeneration:
    """Tests verifying correct IR generation."""

    def test_transfer_generates_shuffle(self) -> None:
        """Transfer generates a shuffle_static operation."""
        tracer = Tracer()
        with tracer:
            x = constant((0,), 42)
            y = transfer(x, to=1)

        graph = tracer.finalize(y)

        # Should have: pcall_static (for constant), shuffle
        opcodes = [op.opcode for op in graph.operations]
        assert (
            "simp.pcall_static" in opcodes
        )  # constant is implemented via pcall_static
        assert "simp.shuffle" in opcodes

    def test_distribute_generates_shuffle_and_converge(self) -> None:
        """Distribute generates shuffle + converge operations."""
        tracer = Tracer()
        with tracer:
            values = [constant((0,), i) for i in range(2)]
            y = distribute(values, frm=0)

        graph = tracer.finalize(y)

        opcodes = [op.opcode for op in graph.operations]
        assert opcodes.count("simp.pcall_static") == 2  # 2 constants
        assert opcodes.count("simp.shuffle") == 2  # 2 shuffles
        assert "simp.converge" in opcodes

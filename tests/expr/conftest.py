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

"""
Shared fixtures for expr tests.
"""

import pytest

from mplang.core.dtype import FLOAT32, INT32, UINT64
from mplang.core.mask import Mask
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType


@pytest.fixture
def pmask_1p():
    """Create a mask for single party (party 0)."""
    return Mask(1)  # 0b1 (party 0 only)


@pytest.fixture
def pmask_2p():
    """Create a mask for two parties (parties 0 and 1)."""
    return Mask(3)  # 0b11 (parties 0 and 1)


@pytest.fixture
def pmask_3p():
    """Create a mask for three parties (parties 0, 1, and 2)."""
    return Mask(7)  # 0b111 (parties 0, 1, and 2)


@pytest.fixture
def pmask_4p():
    """Create a mask for four parties (parties 0, 1, 2, and 3)."""
    return Mask(15)  # 0b1111 (parties 0, 1, 2, and 3)


@pytest.fixture(
    params=[
        Mask(1),  # single party
        Mask(3),  # dual parties
        Mask(7),  # triple parties
        Mask(15),  # quad parties
    ]
)
def pmask_various(request):
    """Parameterized fixture providing different mask configurations."""
    return request.param


@pytest.fixture
def pmask_factory():
    """Factory for creating custom party masks."""

    def _create_pmask(*parties):
        """Create a mask for specified parties.

        Args:
            *parties: Party indices (0, 1, 2, ...)

        Returns:
            Mask: Bitmask with specified parties set
        """
        mask = 0
        for party in parties:
            if party < 0 or party >= 64:  # Reasonable limit for party count
                raise ValueError(f"Party index {party} out of range [0, 63]")
            mask |= 1 << party
        return Mask(mask)

    return _create_pmask


@pytest.fixture
def tensor_info_scalar():
    """Create scalar tensor info."""
    return TensorType(FLOAT32, ())


@pytest.fixture
def tensor_info_1d():
    """Create 1D tensor info."""
    return TensorType(FLOAT32, (5,))


@pytest.fixture
def tensor_info_2d():
    """Create 2D tensor info."""
    return TensorType(FLOAT32, (2, 3))


@pytest.fixture
def tensor_info_3d():
    """Create 3D tensor info."""
    return TensorType(FLOAT32, (2, 3, 4))


@pytest.fixture
def tensor_info_int32():
    """Create INT32 tensor info."""
    return TensorType(INT32, (2, 3))


@pytest.fixture
def tensor_info_uint64():
    """Create UINT64 tensor info."""
    return TensorType(UINT64, (2, 3))


@pytest.fixture(
    params=[
        TensorType(FLOAT32, ()),  # scalar
        TensorType(FLOAT32, (5,)),  # 1D
        TensorType(FLOAT32, (2, 3)),  # 2D
        TensorType(FLOAT32, (2, 3, 4)),  # 3D
        TensorType(INT32, (2, 3)),  # different dtype
        TensorType(UINT64, (2, 3)),  # different dtype
    ]
)
def tensor_info_various(request):
    """Parameterized fixture providing different tensor configurations."""
    return request.param


@pytest.fixture
def tensor_info_factory():
    """Factory for creating custom tensor info."""

    def _create_tensor_info(dtype=FLOAT32, shape=(2, 3)):
        """Create tensor info with specified dtype and shape."""
        return TensorType(dtype, shape)

    return _create_tensor_info


@pytest.fixture
def pfunc_2i1o():
    """Create a mock PFunction for testing."""
    return PFunction(
        fn_type="mock",
        ins_info=[TensorType(FLOAT32, (2, 3)), TensorType(INT32, ())],
        outs_info=[TensorType(FLOAT32, (2, 3))],
        fn_name="mock_func",
    )


@pytest.fixture
def pfunc_1i1o():
    """Create a mock unary PFunction (single input, single output)."""
    return PFunction(
        fn_type="mock",
        ins_info=[TensorType(FLOAT32, (2, 3))],
        outs_info=[TensorType(FLOAT32, (2, 3))],
        fn_name="mock_unary",
    )


@pytest.fixture
def pfunc_2i3o():
    """Create a mock PFunction with multiple outputs."""
    return PFunction(
        fn_type="mock",
        ins_info=[TensorType(FLOAT32, (2, 3)), TensorType(INT32, ())],
        outs_info=[
            TensorType(FLOAT32, (2, 3)),
            TensorType(INT32, ()),
            TensorType(UINT64, (1,)),
        ],
        fn_name="mock_multi_out",
    )


@pytest.fixture
def pfunc_factory():
    """Factory for creating custom PFunction instances."""

    def _create_pfunc(ins_info, outs_info, fn_name="test_func"):
        """Create a PFunction with specified input/output info."""
        return PFunction(
            fn_type="test",
            ins_info=ins_info,
            outs_info=outs_info,
            fn_name=fn_name,
        )

    return _create_pfunc


@pytest.fixture
def test_data_generator():
    """Factory for generating common test data combinations."""

    def _generate(pmask_value=3, dtype=FLOAT32, shape=(2, 3)):
        """Generate test data with specified parameters."""
        import numpy as np

        # Generate sample data based on dtype
        if dtype == FLOAT32:
            data = np.random.randn(*shape).astype(np.float32).tobytes()
        elif dtype == INT32:
            data = np.random.randint(-100, 100, shape, dtype=np.int32).tobytes()
        elif dtype == UINT64:
            data = np.random.randint(0, 1000, shape, dtype=np.uint64).tobytes()
        else:
            data = b""  # Fallback for unknown types

        return {
            "pmask": Mask(pmask_value),
            "tensor_info": TensorType(dtype, shape),
            "data": data,
            "parties": [i for i in range(64) if pmask_value & (1 << i)],
        }

    return _generate

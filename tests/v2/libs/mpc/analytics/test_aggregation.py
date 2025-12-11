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

"""Tests for Homomorphic Aggregation library."""

import numpy as np

import mplang.v2.backends.bfv_impl  # noqa: F401
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.dialects import bfv, tensor
from mplang.v2.libs.mpc.analytics import aggregation


def test_rotate_and_sum():
    # 1. Keygen
    pk, sk = bfv.keygen(poly_modulus_degree=4096)
    gk = bfv.make_galois_keys(sk)
    encoder = bfv.create_encoder(poly_modulus_degree=4096)

    # 2. Data: [1, 2, 3, 4, 0, ...]
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    # Pad to 4096? No, encode handles it.

    t_data = tensor.constant(data)
    pt = bfv.encode(t_data, encoder)
    ct = bfv.encrypt(pt, pk)

    # 3. Aggregate first 4 elements
    # Expected sum: 1+2+3+4 = 10
    res_ct = aggregation.rotate_and_sum(ct, k=4, galois_keys=gk)

    # Decrypt and check
    res_pt = bfv.decrypt(res_ct, sk)
    res_tensor = bfv.decode(res_pt, encoder)

    # res_tensor is an InterpObject.
    # In local simulation, runtime_obj should be the numpy array.
    assert res_tensor.runtime_obj[0] == 10


def test_masked_aggregate():
    pk, sk = bfv.keygen(poly_modulus_degree=4096)
    encoder = bfv.create_encoder(poly_modulus_degree=4096)

    # CT1: [10, 0, 0, ...] -> Mask1: [1, 0, 0, ...]
    # CT2: [0, 20, 0, ...] -> Mask2: [0, 1, 0, ...]

    data1 = np.array([10, 0, 0, 0], dtype=np.int64)
    data2 = np.array([0, 20, 0, 0], dtype=np.int64)

    mask1 = np.array([1, 0, 0, 0], dtype=np.int64)
    mask2 = np.array([0, 1, 0, 0], dtype=np.int64)

    ct1 = bfv.encrypt(bfv.encode(tensor.constant(data1), encoder), pk)
    ct2 = bfv.encrypt(bfv.encode(tensor.constant(data2), encoder), pk)

    pt_mask1 = bfv.encode(tensor.constant(mask1), encoder)
    pt_mask2 = bfv.encode(tensor.constant(mask2), encoder)

    res_ct = aggregation.masked_aggregate([ct1, ct2], [pt_mask1, pt_mask2])

    res_pt = bfv.decrypt(res_ct, sk)
    res = bfv.decode(res_pt, encoder)

    assert res.runtime_obj[0] == 10
    assert res.runtime_obj[1] == 20

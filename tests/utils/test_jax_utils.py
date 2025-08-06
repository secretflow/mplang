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

import jax.numpy as jnp
import pytest

from mplang.utils.jax_utils import HloUtil, jax_jit


def test_hlo():
    def ff(x, y):
        return jnp.cos(x + y)

    a = jnp.array([2])  # a variable on the host
    b = jnp.array([3])  # another variable on the host

    my_ff = jax_jit(ff).trace(a, b)

    mod = my_ff.as_hlo_module()

    text = HloUtil.module_to_text(mod)

    # print(text)

    mod2 = HloUtil.text_to_module(text)

    text2 = HloUtil.module_to_text(mod2)

    # print(text2)

    # print(my_ff.as_hlo_text())
    # print(my_ff.out_info())


def test_parameters():
    # TODO
    pass


def test_return_values():
    def return_none():
        pass

    with pytest.raises(Exception):
        jax_jit(return_none).trace().out_info()

    def return_one():
        return jnp.ones((2, 3))

    ret_list = jax_jit(return_one).trace().out_info()
    assert ret_list.shape[0] == 2

    def return_complicated():
        return {
            "a": jnp.zeros((2, 3)),
            "b": [
                jnp.zeros((2, 3)),
                jnp.zeros((2, 3)),
            ],
            "c": {
                "a": jnp.ones((2, 3)),
                "b": jnp.ones((2, 3)),
            },
        }

    # print(jax_jit(return_complicated).trace().out_info())
    ret_dict = jax_jit(return_complicated).trace().out_info()
    assert isinstance(ret_dict, dict)
    assert "a" in ret_dict
    assert "b" in ret_dict
    assert len(ret_dict["b"]) == 2
    assert "c" in ret_dict


def test_hlo_xla():
    def ff(x, y):
        return jnp.cos(x + y)

    a = jnp.array([2])
    b = jnp.array([3])

    mod = jax_jit(ff).trace(a, b).as_hlo_module()

    # check baseline
    text = HloUtil.module_to_text(mod)

    # m2t = m2p | p2t
    prot = HloUtil.module_to_proto(mod)
    text0 = HloUtil.proto_to_text(prot)
    assert text0 == text

    mlir0 = HloUtil.proto_to_mlir(prot)
    mlir = HloUtil.text_to_mlir(text)

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

import json
from typing import Any
import mplang
import mplang.simp as simp
from mplang.core import mpir
from mplang.core.cluster import ClusterSpec, PhysicalNode, LogicalDevice, RuntimeInfo
from mplang.api import CompileOptions
from google.protobuf import json_format
from jax.tree_util import PyTreeDef, tree_flatten

from mplang.frontend.base import FEOp
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType
from mplang.core.dtype import UINT8 # Corrected import

# --- Define Mock TEE operations as Frontend Operations (FEOps) ---

# Define some placeholder types for quotes and keys. We'll treat them as byte arrays.
quote_type = TensorType(UINT8, (1,))
key_type = TensorType(UINT8, (1,))
encrypted_type = TensorType(UINT8, (1,))

class QuoteGenOp(FEOp):
    def __call__(self, num_quotes: int = 1) -> tuple[PFunction, list[Any], PyTreeDef]:
        pfunc = PFunction(
            fn_type="tee.quote_gen",
            ins_info=(),
            outs_info=(quote_type, key_type),
            attrs={'num_quotes': num_quotes}
        )
        _, treedef = tree_flatten((quote_type, key_type))
        return pfunc, [], treedef

class QuoteVerifyOp(FEOp):
    def __call__(self, quote: Any) -> tuple[PFunction, list[Any], PyTreeDef]:
        pfunc = PFunction(
            fn_type="tee.quote_verify",
            ins_info=(quote_type,),
            outs_info=(key_type,),
        )
        _, treedef = tree_flatten(key_type)
        return pfunc, [quote], treedef

class EncryptOp(FEOp):
    def __call__(self, data: Any, key: Any) -> tuple[PFunction, list[Any], PyTreeDef]:
        pfunc = PFunction(
            fn_type="tee.symmetric_encrypt",
            ins_info=(TensorType.from_obj(data), key_type),
            outs_info=(encrypted_type,),
        )
        _, treedef = tree_flatten(encrypted_type)
        return pfunc, [data, key], treedef

quote_gen = QuoteGenOp()
quote_verify = QuoteVerifyOp()
symmetric_encrypt = EncryptOp()

# Parties: 0, 1 are data providers; 2 is TEE party
P0, P1, P2 = 0, 1, 2

@mplang.function
def secure_sum_with_attestation():
    quote, key = simp.runAt(P2, quote_gen)(1)
    quote_at_p0 = simp.p2p(P2, P0, quote)
    verified_key = simp.runAt(P0, quote_verify)(quote_at_p0)
    data_a = simp.prandint(0, 100, party=P0)
    data_b = simp.prandint(0, 100, party=P1)
    encrypted_a = simp.runAt(P0, symmetric_encrypt)(data_a, verified_key)
    data_a_at_tee = simp.p2p(P0, P2, encrypted_a)
    data_b_at_tee = simp.p2p(P1, P2, data_b)
    return simp.reveal(data_a_at_tee), simp.reveal(data_b_at_tee)

if __name__ == "__main__":
    world_size = 3
    nodes = { f"node{i}": PhysicalNode(name=f"node{i}", rank=i, endpoint=f"localhost:{5000+i}", runtime_info=RuntimeInfo(version="sim", platform="sim", backends=["__all__"])) for i in range(world_size) }
    devices = {
        "SPU": LogicalDevice(name="SPU", kind="SPU", members=list(nodes.values()), config={"protocol": "SEMI2K", "field": "FM128"}),
        "TEE": LogicalDevice(name="TEE", kind="TEE", members=[nodes["node2"]])
    }
    cluster_spec = ClusterSpec(nodes=nodes, devices=devices)
    copts = CompileOptions(cluster_spec)
    compiled = mplang.compile(copts, secure_sum_with_attestation)
    ir_writer = mpir.Writer()
    func_expr = compiled.make_expr()
    graph_proto = ir_writer.dumps(func_expr)
    json_dict = json_format.MessageToDict(graph_proto)
    output_filename = "test_graph.json"
    with open(output_filename, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"Successfully generated MPIR graph and saved it to '{output_filename}'")
    print("You can now load this file into the visualizer tool.")

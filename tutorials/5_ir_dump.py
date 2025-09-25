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


import mplang
import mplang.simp as simp
from mplang.core import mpir


@mplang.function
def millionaire():
    # Note: mpl.run(random.randint) will not work, because
    # the random number generator's state is captured and will always
    # return the same number on both parties.
    x = simp.prandint(0, 10)
    y = simp.prandint(0, 10)

    # both of them seal it
    x_ = simp.sealFrom(x, 0)
    y_ = simp.sealFrom(y, 1)

    # compare it seally.
    z_ = simp.srun(lambda x, y: x < y)(x_, y_)

    # reveal it to all.
    z = simp.reveal(z_)

    return x, y, z


if __name__ == "__main__":
    world_size = 4

    # Create compilation options for ahead-of-time (AOT) compilation
    # This specifies the number of parties and which parties have SPU devices
    copts = mplang.CompileOptions.simple(world_size)

    # Compile the function to get the IR representation
    # This traces the function and creates a static computation graph
    compiled = mplang.compile(copts, millionaire)
    print("Compiled function:", compiled)
    print()

    # Get the compiler IR (MPIR representation)
    # This shows the low-level multi-party computation instructions
    compiled_ir = compiled.compiler_ir()
    print("IR proto (MPIR representation):")
    print(compiled_ir)
    print()

    # You can also examine the structure of the traced function
    print(f"Function name: {compiled.func_name}")
    print(f"Input variables: {len(compiled.in_vars)}")
    print(f"Output variables: {len(compiled.out_vars)}")
    print(f"Captured variables: {len(compiled.capture_map)}")

    print("=" * 60)
    print("Demonstration of IR Writer - Raw Protobuf Text Format")
    print("=" * 60)

    # Use the IR Writer to serialize the expression tree to protobuf
    ir_writer = mpir.Writer()
    func_expr = compiled.make_expr()
    graph_proto = ir_writer.dumps(func_expr)

    print("Raw GraphProto in text format:")
    print(graph_proto)
    print()

    # Use the new statistics function from mpir module
    print(mpir.get_graph_statistics(graph_proto))

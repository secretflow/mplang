"""Example: pcall + jax_fn integration.

Demonstrates how SIMP dialect (pcall_static) can reference Tensor dialect (jax_fn).
Uses EDSL printer to display the computation graph.
"""

import jax.numpy as jnp
import numpy as np

import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang2.dialects.simp import pcall_static
from mplang2.dialects.tensor import jax_fn


def main():
    """Demonstrate pcall + jax_fn with graph printing."""

    # Define native JAX functions
    def square(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.square(x)

    def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.add(x, y)

    # Prepare input data (MP types)
    x = el.InterpObject(
        np.array([1.0, 2.0, 3.0]),
        elt.MPType(elt.TensorType(elt.f32, (3,)), (0,)),  # P0 holds
    )

    y = el.InterpObject(
        np.array([4.0, 5.0, 6.0]),
        elt.MPType(elt.TensorType(elt.f32, (3,)), (0,)),  # P0 holds
    )

    # Define computation using pcall + jax_fn
    def compute(x, y):
        # P0 executes JAX square computation
        squared = pcall_static((0,), jax_fn(square), x)
        # P0 executes JAX add computation
        result = pcall_static((0,), jax_fn(add), squared, y)
        return result

    # Trace to generate graph
    traced = el.trace(compute, x, y)

    # Print computation graph using EDSL printer
    print("=" * 70)
    print("Computation Graph")
    print("=" * 70)
    print(el.format_graph(traced.graph))
    print()

    # Print output type
    print("=" * 70)
    print("Output Information")
    print("=" * 70)
    print(f"Type: {traced.graph.outputs[0].type}")
    print("Interpretation: Party 0 holds a float32[3] Tensor")
    print()

    # Print detailed operation information
    print("=" * 70)
    print("Operation Details")
    print("=" * 70)
    for i, op in enumerate(traced.graph.operations):
        print(f"\nOperation {i}: {op.opcode}")
        print(f"  Attributes: {op.attrs}")
        print(f"  Inputs: {len(op.inputs)}")
        print(f"  Outputs: {len(op.outputs)}")
        print(f"  Regions: {len(op.regions)}")

        if op.regions:
            print("\n  Region content:")
            region_str = el.format_graph(op.regions[0])
            # Indent each line of the region output
            for line in region_str.split("\n"):
                print(f"    {line}")


if __name__ == "__main__":
    main()

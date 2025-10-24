#!/usr/bin/env python3
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

"""
Tutorial: Auto Device Inference

This tutorial demonstrates the new auto device inference feature in MPLang.

Key Features:
1. Automatic device inference from function arguments
2. Backward compatible with explicit device specification
3. Smart device selection (prefers SPU/TEE over PPU for mixed-device scenarios)
"""

import mplang as mp


def example_1_explicit_device():
    """Original explicit device specification (still supported)."""
    print("\n=== Example 1: Explicit Device (Original Syntax) ===")

    # Use simple simulator with 2 parties (creates P0, P1, SP0 devices)
    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        # Explicit device specification - original syntax
        x = mp.device("P0")(lambda: 10)()
        y = mp.device("P0")(lambda a: a + 5)(x)
        return y

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    print(f"Result: {int(fetched[0])}")  # 15


def example_2_auto_device_same_ppu():
    """Auto device inference when all args are on the same PPU."""
    print("\n=== Example 2: Auto Device - Same PPU ===")

    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        # Create data on P0
        x = mp.device("P0")(lambda: 10)()
        y = mp.device("P0")(lambda: 5)()

        # Auto device inference - will run on P0 because both x and y are on P0
        z = mp.device(lambda a, b: a + b)(x, y)
        return z

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    print(f"Result: {int(fetched[0])}")  # 15
    print("Device was automatically inferred as P0 from arguments")


def example_3_decorator_style():
    """Using @device decorator with auto inference."""
    print("\n=== Example 3: Decorator Style with Auto Device ===")

    sim = mp.Simulator.simple(2)

    # Define a reusable function with auto device
    @mp.device
    def add(a, b):
        return a + b

    @mp.device
    def multiply(a, b):
        return a * b

    @mp.function
    def compute():
        # Create initial data on P0
        x = mp.device("P0")(lambda: 3)()
        y = mp.device("P0")(lambda: 4)()

        # Auto device inference for both operations
        sum_val = add(x, y)  # Infers P0
        product = multiply(x, y)  # Infers P0

        # Combine results
        final = add(sum_val, product)  # Infers P0
        return final

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    print(f"Result: (3+4) + (3*4) = {int(fetched[0])}")  # 19
    print("All operations automatically inferred device from arguments")


def example_4_inline_usage():
    """Inline lambda usage with auto device."""
    print("\n=== Example 4: Inline Lambda with Auto Device ===")

    sim = mp.Simulator.simple(2)

    @mp.function
    def compute():
        x = mp.device("P0")(lambda: 20)()

        # Inline lambda with auto device inference
        y = mp.device(lambda a: a * 2)(x)
        z = mp.device(lambda a: a + 10)(y)

        return z

    result = mp.evaluate(sim, compute)
    fetched = mp.fetch(sim, result)
    print(f"Result: (20 * 2) + 10 = {int(fetched[0])}")  # 50
    print("All inline lambdas automatically used P0")


def example_5_error_cases():
    """Demonstrate error cases that require explicit device."""
    print("\n=== Example 5: Error Cases ===")

    sim = mp.Simulator.simple(2)

    print("\n1. No device objects in arguments:")

    @mp.function
    def compute_no_args():
        # This will fail - no device objects to infer from
        return mp.device(lambda: 42)()

    try:
        mp.evaluate(sim, compute_no_args)
    except ValueError as e:
        print(f"   ✗ Error (expected): {e}")

    print("\n2. Explicit device always works:")

    @mp.function
    def compute_explicit():
        # This works - explicit device specified
        return mp.device("P0")(lambda: 42)()

    result = mp.evaluate(sim, compute_explicit)
    fetched = mp.fetch(sim, result)
    print(f"   ✓ Success: {int(fetched[0])}")


def main():
    """Run all examples."""
    print("=" * 70)
    print("MPLang Auto Device Inference Tutorial")
    print("=" * 70)

    example_1_explicit_device()
    example_2_auto_device_same_ppu()
    example_3_decorator_style()
    example_4_inline_usage()
    example_5_error_cases()

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Use device('P0')(fn) for explicit device specification")
    print("  - Use device(fn) or @device for automatic inference")
    print("  - Auto inference works when all args are on the same device")
    print("  - Explicit device specification is required when inference fails")
    print("=" * 70)


if __name__ == "__main__":
    main()

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

"""Tutorial 08: Initialize NNX Models on Devices and Pass State as Dicts

Demonstrates the proper pattern for using Flax NNX with MPLang v1:
1. Initialize model on device and return state as pure Python dict
2. Pass the state dict across @mp.function boundaries
3. Reconstruct model on device using the state dict for inference
4. Train models with optimizer state management

Key insight: NNX State can be converted to/from pure Python dicts using:
- state.to_pure_dict() → returns dict with format {layer: {param: [array, None]}}
- state.replace_by_pure_dict(dict) → reconstructs state from this format

Optimizer state can also be managed as pure Python dicts, enabling:
- Stateful training across @mp.function boundaries
- Checkpoint-style model updates
- No need to store graphdef (can be reconstructed from model class)

Usage:
    uv run python tutorials/v1/device/08_initialize_nnx_on_device.py
"""

import jax
import jax.numpy as jnp

import mplang.v1 as mp
import optax
from flax import nnx

# ============================================================================
# Section 1: Pure Python Functions (Model Definition & Logic)
# ============================================================================


class SimpleMLP(nnx.Module):
    """Simple Multi-Layer Perceptron for demonstration."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs
    ):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


def init_model_logic(input_dim: int, hidden_dim: int, output_dim: int, seed: int):
    """Pure function: Initialize model and return state as dict.

    This is the core logic without MPLang decoration.

    Args:
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed for initialization

    Returns:
        Pure Python dict containing model parameters
    """
    # Create model
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        rngs=nnx.Rngs(seed),
    )

    # Split and convert state to pure dict
    _graphdef, state = nnx.split(model)
    state_dict = state.to_pure_dict()

    print(f"[Device P0] Initialized model with {len(state_dict)} parameter groups")
    print(f"[Device P0] State dict keys: {list(state_dict.keys())}")

    # Return the state dict - it's in format {layer: {param: [array, None]}}
    return state_dict


def inference_logic(
    x: jax.Array,
    params_dict: dict,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
):
    """Pure function: Run inference using state dict.

    This is the core logic without MPLang decoration.

    Args:
        x: Input data [batch_size, input_dim]
        params_dict: Pure Python dict containing model parameters
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed (for GraphDef reconstruction)

    Returns:
        Model output logits
    """
    # Create an abstract model to get the GraphDef.
    # nnx.eval_shape only creates the abstract structure; parameter arrays are
    # materialized later when replace_by_pure_dict is called with params_dict.
    abs_model = nnx.eval_shape(
        lambda: SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rngs=nnx.Rngs(seed),
        )
    )
    graphdef, abs_state = nnx.split(abs_model)

    print(f"[Device P0] Received state dict with keys: {list(params_dict.keys())}")

    # The params_dict is already in the format from to_pure_dict(): {key: [array, None]}
    # So we can use it directly with replace_by_pure_dict
    abs_state.replace_by_pure_dict(params_dict)

    print("[Device P0] Reconstructed state from dict")

    # Merge to get working model with the passed parameters
    model = nnx.merge(graphdef, abs_state)

    print("[Device P0] Reconstructed model from state dict")
    print("[Device P0] Running inference...")

    # Run inference
    output = model(x)

    print("[Device P0] Inference complete!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    return output


def init_model_with_optimizer_logic(
    input_dim: int, hidden_dim: int, output_dim: int, seed: int, learning_rate: float
):
    """Pure function: Initialize model + optimizer and return states as dict.

    This is the core logic without MPLang decoration.

    Args:
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed for initialization
        learning_rate: Learning rate for optimizer

    Returns:
        Dict with model_state_dict, opt_state, and step
    """
    # Create model
    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        rngs=nnx.Rngs(seed),
    )

    # Split and convert model state to pure dict
    _graphdef, state = nnx.split(model)
    model_state_dict = state.to_pure_dict()

    # Initialize optimizer and convert its state to pure dict
    # Note: optax states are already pytrees, but we'll store them explicitly
    tx = optax.sgd(learning_rate)
    opt_state = tx.init(model_state_dict)

    print("[Device P0] Initialized model with optimizer")
    print(f"[Device P0] Model state keys: {list(model_state_dict.keys())}")
    print(f"[Device P0] Optimizer state type: {type(opt_state)}")

    return {
        "model_state_dict": model_state_dict,
        "opt_state": opt_state,
        "step": 0,
    }


def train_step_logic(
    train_state: dict,
    batch_x: jax.Array,
    batch_y: jax.Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
    learning_rate: float,
):
    """Pure function: Perform one training step.

    This is the core logic without MPLang decoration.
    Forward → loss → backward → update

    Args:
        train_state: Dict with model_state_dict, opt_state, step
        batch_x: Input batch (batch_size, input_dim)
        batch_y: Target batch (batch_size, output_dim)
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed (for GraphDef reconstruction)
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (updated_train_dict, loss_value)
    """
    # Extract state
    model_state_dict = train_state["model_state_dict"]
    opt_state = train_state["opt_state"]
    step = train_state["step"]

    # Define loss function that works with jax.grad
    def loss_fn(state_dict, x, y):
        # Reconstruct model from state dict for forward pass.
        # Create abstract model to get GraphDef; parameters are then materialized from state_dict.
        abs_model = nnx.eval_shape(
            lambda: SimpleMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                rngs=nnx.Rngs(seed),
            )
        )
        graphdef, abs_state = nnx.split(abs_model)
        abs_state.replace_by_pure_dict(state_dict)
        model = nnx.merge(graphdef, abs_state)

        # Forward pass
        logits = model(x)

        # MSE loss
        loss = jnp.mean((logits - y) ** 2)
        return loss

    # Compute loss and gradients in a single pass (efficient!)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss_value, grads_dict = loss_and_grad_fn(model_state_dict, batch_x, batch_y)

    # Apply optimizer update
    tx = optax.sgd(learning_rate)
    updates, new_opt_state = tx.update(grads_dict, opt_state, model_state_dict)
    new_model_state_dict = optax.apply_updates(model_state_dict, updates)

    # Return updated state
    new_train_dict = {
        "model_state_dict": new_model_state_dict,
        "opt_state": new_opt_state,
        "step": step + 1,
    }

    return new_train_dict, loss_value


# ============================================================================
# Section 2: MPLang Functions (Multi-Party Device Execution)
# ============================================================================

# Cluster configuration
cluster_spec = mp.ClusterSpec.from_dict(
    {
        "nodes": [
            {"name": "node_0", "endpoint": "127.0.0.1:61920"},
            {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["node_0", "node_1"],
                "config": {"protocol": "SEMI2K", "field": "FM128"},
            },
            "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        },
    }
)


@mp.function
def demo_basic_model_init_and_inference(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
    test_input: jax.Array,
):
    """Combined MPLang function: Initialize model, run inference, return both.

    This combines two operations into one MPLang function to reduce overhead:
    1. Initialize model and get state dict
    2. Run inference with the state dict

    Args:
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed for initialization
        test_input: Input data for inference

    Returns:
        Tuple of (state_dict, inference_output)
    """

    def _combined():
        # Step 1: Initialize model
        state_dict = init_model_logic(input_dim, hidden_dim, output_dim, seed)

        # Step 2: Run inference using the same state dict
        output = inference_logic(
            test_input, state_dict, input_dim, hidden_dim, output_dim, seed
        )

        return state_dict, output

    return mp.device("P0", fe_type="nnx")(_combined)()


@mp.function
def train_model_for_n_steps(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
    learning_rate: float,
    x_train: jax.Array,
    y_train: jax.Array,
    n_steps: int,
):
    """Combined MPLang function: Initialize model with optimizer and train for N steps.

    This combines the entire training pipeline into one MPLang function:
    1. Initialize model with optimizer
    2. Run N training steps in a loop
    3. Return final state and loss history

    This is much more efficient than calling train_step N times across @mp.function boundaries!

    Args:
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed for initialization
        learning_rate: Learning rate for optimizer
        x_train: Training input data
        y_train: Training target data
        n_steps: Number of training steps to run

    Returns:
        Tuple of (final_train_dict, loss_history)
    """

    def _train_loop():
        # Initialize model with optimizer
        train_dict = init_model_with_optimizer_logic(
            input_dim, hidden_dim, output_dim, seed, learning_rate
        )

        # Training loop - all steps run on device without crossing boundaries
        loss_history = []
        for _step_idx in range(n_steps):
            train_dict, loss_value = train_step_logic(
                train_dict,
                x_train,
                y_train,
                input_dim,
                hidden_dim,
                output_dim,
                seed,
                learning_rate,
            )
            loss_history.append(loss_value)

        return train_dict, jnp.array(loss_history)

    return mp.device("P0", fe_type="nnx")(_train_loop)()


# ============================================================================
# Section 3: Main Execution Flow
# ============================================================================


def main():
    """Main demonstration: Shows efficient batched operations for NNX models."""
    print("=" * 80)
    print("NNX Model with State Dict Pattern")
    print("=" * 80)

    # Setup simulator
    simulator = mp.Simulator(cluster_spec)

    # =========================================================================
    # Efficient Batched Operations (RECOMMENDED)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Efficient Batched MPLang Functions (RECOMMENDED)")
    print("=" * 80)
    print("Combining multiple operations into single @mp.function calls")
    print("reduces overhead and improves performance!\n")

    # Step 1: Combined init + inference in ONE MPLang call
    print("[Step 1] Initialize model + Run inference (combined operation)...")

    # Create test input
    test_key = jax.random.PRNGKey(999)
    test_input = jax.random.normal(test_key, (4, 8))

    combined_result = mp.evaluate(
        simulator,
        demo_basic_model_init_and_inference,
        input_dim=8,
        hidden_dim=16,
        output_dim=2,
        seed=42,
        test_input=test_input,
    )

    state_dict, output = mp.fetch(simulator, combined_result)

    # Handle wrapped output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        output = output[0]
    if not isinstance(output, jnp.ndarray):
        output = jnp.array(output)

    print("✅ Model initialized and inference complete!")
    print(f"   State dict keys: {list(state_dict.keys())}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0]}")

    # Step 2: Combined training loop in ONE MPLang call
    print("\n[Step 2] Initialize + Train for 10 steps (combined operation)...")
    print("This is MUCH more efficient than 10 separate @mp.function calls!\n")

    # Create training data
    x_train = jax.random.normal(jax.random.PRNGKey(888), (32, 8))
    true_weights = jnp.array(
        [
            [2.0],
            [-1.0],
            [0.5],
            [1.5],
            [-0.5],
            [1.0],
            [-1.5],
            [0.0],
        ]
    )
    true_weights2 = jnp.array([[1.0], [-0.5]])
    y_train = (
        x_train @ true_weights @ true_weights2.T
        + jax.random.normal(jax.random.PRNGKey(999), (32, 2)) * 0.1
    )

    train_result = mp.evaluate(
        simulator,
        train_model_for_n_steps,
        input_dim=8,
        hidden_dim=16,
        output_dim=2,
        seed=42,
        learning_rate=0.01,
        x_train=x_train,
        y_train=y_train,
        n_steps=10,
    )

    final_train_dict, loss_history = mp.fetch(simulator, train_result)

    # Handle wrapped arrays
    if isinstance(loss_history, (list, tuple)):
        loss_history = loss_history[0] if len(loss_history) > 0 else loss_history
    if not isinstance(loss_history, jnp.ndarray):
        loss_history = jnp.array(loss_history)

    print("\n✅ Training complete!")
    print(f"   Final step: {final_train_dict['step']}")
    print(f"   Initial loss: {loss_history[0]:.4f}")
    print(f"   Final loss: {loss_history[-1]:.4f}")
    print(f"   Loss reduction: {(loss_history[0] - loss_history[-1]):.4f}")
    print(
        f"   All losses: {[f'{loss:.4f}' for loss in loss_history[:5]]}... (showing first 5)"
    )

    # Summary
    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("\n✅ NNX + MPLANG PATTERNS:")
    print("1. ✅ Separate pure Python logic from MPLang decorators")
    print("   - Section 1: Pure functions (testable, reusable)")
    print("   - Section 2: MPLang wrappers (device placement)")
    print("   - Section 3: Main execution (orchestration)")

    print("\n2. ✅ Batch operations into larger @mp.function calls")
    print("   - demo_basic_model_init_and_inference: init + inference together")
    print("   - train_model_for_n_steps: init + N training steps together")
    print("   - Reduces @mp.function boundary crossings = better performance!")

    print("\n3. ✅ Use state dicts for cross-boundary communication")
    print("   - state.to_pure_dict() → serialize model state")
    print("   - state.replace_by_pure_dict() → deserialize model state")
    print("   - Works with optimizer state too!")

    print("\n4. ✅ Use nnx.eval_shape() for GraphDef reconstruction")
    print("   - Creates abstract model structure for obtaining GraphDef")
    print("   - Actual parameters are materialized when replace_by_pure_dict is called")

    print("\n5. ✅ Use jax.value_and_grad for efficient gradient computation")
    print("   - Computes loss and gradients in a single pass")
    print("   - More efficient than separate forward and backward passes")

    print("\n🎯 BEST PRACTICES:")
    print("   • Batch related operations into single @mp.function calls")
    print("   • Minimize @mp.function boundary crossings")
    print("   • Group operations together for efficiency")
    print("   • Use state dicts for flexible model management")
    print("   • Leverage nnx.eval_shape() for memory-efficient model reconstruction")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

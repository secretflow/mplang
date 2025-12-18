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
import optax
from flax import nnx

import mplang.v1 as mp


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


# Cluster configuration
cluster_spec = mp.ClusterSpec.from_dict({
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
})


@mp.function
def initialize_model_and_return_state_dict(
    input_dim: int, hidden_dim: int, output_dim: int, seed: int
):
    """Initialize model on device P0 and return state as a pure dict.

    Returns:
        Pure Python dict containing model parameters (can cross @mp.function boundaries)
    """

    def _init():
        # Create model on device
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

    return mp.device("P0", fe_type="nnx")(_init)()


@mp.function
def run_inference_with_state_dict(
    test_input: jax.Array,
    state_dict: dict,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
):
    """Run inference using a state dict passed from outside.

    Args:
        test_input: Input data [batch_size, input_dim]
        state_dict: Pure Python dict containing model parameters
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed (for GraphDef reconstruction)

    Returns:
        Model output logits
    """

    def _infer(x, params_dict):
        """Inner function that takes explicit inputs."""
        # Create an abstract model to get the GraphDef (memory efficient!)
        # This only creates the structure without allocating actual arrays
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

    return mp.device("P0", fe_type="nnx")(_infer)(test_input, state_dict)


@mp.function
def initialize_model_with_optimizer(
    input_dim: int, hidden_dim: int, output_dim: int, seed: int, learning_rate: float
):
    """Initialize model + optimizer on device P0 and return both states as dict.

    Returns:
        Dict with model_state_dict and opt_state (both as pure Python dicts)
    """

    def _init():
        # Create model on device
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

    return mp.device("P0", fe_type="nnx")(_init)()


@mp.function
def train_step(
    train_dict: dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
    learning_rate: float,
):
    """Perform one training step: forward → loss → backward → update.

    Args:
        train_dict: Dict with model_state_dict, opt_state, step
        x: Input batch (batch_size, input_dim)
        y: Target batch (batch_size, output_dim)
        input_dim: Model input dimension
        hidden_dim: Model hidden dimension
        output_dim: Model output dimension
        seed: Random seed (for GraphDef reconstruction)
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (updated_train_dict, loss_value)
    """

    def _train(train_state, batch_x, batch_y):
        # Extract state
        model_state_dict = train_state["model_state_dict"]
        opt_state = train_state["opt_state"]
        step = train_state["step"]

        # Define loss function that works with jax.grad
        def loss_fn(state_dict, x, y):
            # Reconstruct model from state dict for forward pass
            # Create abstract model to get GraphDef (memory efficient!)
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

        # Compute gradients
        loss_value = loss_fn(model_state_dict, batch_x, batch_y)
        grad_fn = jax.grad(loss_fn)
        grads_dict = grad_fn(model_state_dict, batch_x, batch_y)

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

    return mp.device("P0", fe_type="nnx")(_train)(train_dict, x, y)


def main():
    """Main demonstration: initialize model on P0, then run inference using state dict."""
    print("=" * 80)
    print("NNX Model with State Dict Pattern")
    print("=" * 80)

    # Setup simulator
    simulator = mp.Simulator(cluster_spec)

    # Step 1: Initialize model on device P0 and get state as dict
    print("\n[Step 1] Initializing model on device P0...")
    state_dict_result = mp.evaluate(
        simulator,
        initialize_model_and_return_state_dict,
        input_dim=8,
        hidden_dim=16,
        output_dim=2,
        seed=42,
    )

    # Fetch the state dict to driver
    state_dict = mp.fetch(simulator, state_dict_result)
    print(f"\n[Driver] Received state dict with keys: {list(state_dict.keys())}")

    # Step 2: Run inference using the state dict
    print("\n[Step 2] Running inference with state dict...")

    # Create test input
    test_key = jax.random.PRNGKey(999)
    test_input = jax.random.normal(test_key, (4, 8))

    output_result = mp.evaluate(
        simulator,
        run_inference_with_state_dict,
        test_input,
        state_dict_result,
        input_dim=8,
        hidden_dim=16,
        output_dim=2,
        seed=42,  # Same seed to get same GraphDef
    )

    output = mp.fetch(simulator, output_result)

    # Handle output if it's wrapped
    if isinstance(output, (list, tuple)) and len(output) > 0:
        output = output[0]
    if not isinstance(output, jnp.ndarray):
        output = jnp.array(output)

    print("\n[Driver] Inference complete!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")

    # Step 3: Initialize model with optimizer
    print("\n[Step 3] Initializing model with optimizer on device P0...")
    train_state_dict_result = mp.evaluate(
        simulator,
        initialize_model_with_optimizer,
        input_dim=8,
        hidden_dim=16,
        output_dim=2,
        seed=42,
        learning_rate=0.01,
    )

    # Fetch the train state dict to driver
    train_state_dict = mp.fetch(simulator, train_state_dict_result)
    print(
        f"\n[Driver] Received train state dict with keys: {list(train_state_dict.keys())}"
    )

    # Step 4: Perform a single training step
    print("\n[Step 4] Performing a single training step...")

    # Create a simple learnable dataset: y = 2*x + noise
    # This gives the model something real to learn
    x_train = jax.random.normal(jax.random.PRNGKey(888), (32, 8))
    # Create target with a simple linear relationship
    true_weights = jnp.array([
        [2.0],
        [-1.0],
        [0.5],
        [1.5],
        [-0.5],
        [1.0],
        [-1.5],
        [0.0],
    ])
    true_weights2 = jnp.array([[1.0], [-0.5]])
    y_train = (
        x_train @ true_weights @ true_weights2.T
        + jax.random.normal(jax.random.PRNGKey(999), (32, 2)) * 0.1
    )

    train_result = mp.evaluate(
        simulator,
        train_step,
        train_state_dict_result,
        x_train,
        y_train,
        input_dim=8,
        hidden_dim=16,
        output_dim=2,
        seed=42,
        learning_rate=0.01,
    )

    # Fetch only for human inspection
    updated_train_dict, loss_value = mp.fetch(simulator, train_result)
    print("\n[Driver] Training step complete!")
    print(f"  Loss value: {loss_value}")
    print(
        f"  Updated model state keys: {list(updated_train_dict['model_state_dict'].keys())}"
    )
    print(f"  Updated optimizer state type: {type(updated_train_dict['opt_state'])}")
    print(f"  Step number: {updated_train_dict['step']}")

    # Step 5: Demonstrate Optimizer State Persists Across Multiple Training Steps
    print("\n" + "=" * 80)
    print("[Step 5] Optimizer State Persists Across MPLang Function Boundaries")
    print("=" * 80)
    print("Demonstrating that optimizer state can be passed back and forth")
    print("\nRunning 5 more training steps with THE SAME batch...")
    print("Note: Loss should decrease steadily as the model learns the pattern!\n")

    # Keep using train_result (before fetch) for subsequent computations
    current_train_result = train_result

    for step_idx in range(5):
        # THE KEY POINT: Pass train_result[0] (the train_dict) to next mp.evaluate!
        # train_result is a tuple (train_dict, loss), we need just the train_dict
        # Use the SAME training data to show actual learning
        current_train_result = mp.evaluate(
            simulator,
            train_step,
            current_train_result[
                0
            ],  # Pass just the train_dict (first element of tuple)
            x_train,  # Same data - so we can see the model actually learning!
            y_train,
            input_dim=8,
            hidden_dim=16,
            output_dim=2,
            seed=42,
            learning_rate=0.01,
        )

        # Fetch only for display (human inspection)
        updated_train_dict, loss_value = mp.fetch(simulator, current_train_result)
        print(
            f"  Step {step_idx + 2}: Loss = {loss_value}, Global step = {updated_train_dict['step']}"
        )

    # Final fetch for summary
    final_train_dict, final_loss = mp.fetch(simulator, current_train_result)

    print("\n" + "=" * 80)
    print("Training Progress Summary")
    print("=" * 80)
    print(f"Final loss: {final_loss}")
    print(f"Final optimizer state type: {type(final_train_dict['opt_state'])}")
    print(f"Final global step: {final_train_dict['step']}")

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("1. ✅ Initialize NNX models on devices with mp.device('P0', fe_type='nnx')")
    print("2. ✅ Convert state to pure dict: state.to_pure_dict()")
    print("   - Format: {layer: {param: [array, None]}}")
    print("3. ✅ Pass dict across @mp.function boundaries (works in MPLang v1!)")
    print("4. ✅ Reconstruct state: state.replace_by_pure_dict(state_dict)")
    print("5. ✅ Use nnx.eval_shape() for memory-efficient GraphDef creation")
    print("   - Creates abstract model without allocating arrays!")
    print("6. ✅ Merge with GraphDef: model = nnx.merge(graphdef, state)")
    print("7. ✅ This enables efficient model reuse without reinitializing weights!")
    print("8. ✅ Initialize model with optimizer state management")
    print("   - Use optax for flexible optimizer integration")
    print("   - Manage optimizer state as pure Python dicts")
    print(
        "9. ✅ Perform training steps with full control: forward, loss, backward, update"
    )
    print("   - Stateful training across @mp.function boundaries")
    print("   - Checkpoint-style updates with pure dicts")
    print(
        "10. ✅ CRITICAL: Optimizer state persists across MPLang function boundaries!"
    )
    print("    - Pass train_dict (with opt_state) between training steps")
    print("    - Optimizer accumulates state (momentum, etc.) correctly")
    print("    - Step counter tracks global training progress")
    print("    - Enables iterative training with stateful optimizers")
    print("    - Loss decreases steadily when training on same batch!")
    print("\nThis pattern solves the MPLang v1 limitation of passing complex types")
    print("by using pure Python dicts that can cross @mp.function boundaries.")
    print("Optimizer state management enables proper multi-step training!")
    print("=" * 80)


if __name__ == "__main__":
    main()

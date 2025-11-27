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

"""Device: Multi-Step Training with NNX Models

A focused tutorial demonstrating multi-step neural network training using:
- NNX MLP for binary classification
- Multi-step training with jax.lax.fori_loop
- Complete training loop with loss tracking and convergence analysis
- Proper gradient computation and parameter updates
- Stateful models with batch normalization

Learning objectives:
1. Build both stateless and stateful NNX MLPs for binary classification
2. Implement multi-step training with jax.lax.fori_loop
3. Track training progress with loss visualization
4. Compare stateless vs stateful model performance
5. Demonstrate proper state management in distributed training
6. Use NNX Functional API for distributed computation

Key concepts:
- NNX Models: Modern neural network abstraction in Flax
- NNX Functional API: nnx.split/merge for stateless computation boundaries
- Multi-step training: Efficient training loops using JAX transformations
- Stateful training: Batch normalization for improved convergence
- Loss tracking: Monitoring training progress across multiple iterations
- Gradient computation: jax.value_and_grad for backpropagation
- Distributed constraints: Working within MPLang execution limitations
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import mplang.v1 as mp

# JAX will use float32 by default, which is efficient and sufficient for most ML tasks


class MultiStepMLP(nnx.Module):
    """Multi-Layer Perceptron designed for multi-step training demonstrations.

    This model demonstrates how to handle neural network training in distributed
    settings using the NNX Functional API. It focuses on the essential aspects
    of multi-step training without the complexity of stateful components.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs
    ):
        """Initialize the MLP.

        Args:
            input_dim: Size of input features
            hidden_dim: Size of hidden layer
            output_dim: Size of output layer (number of classes)
            rngs: Random number generators for initialization
        """
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool = True) -> jax.Array:
        """Forward pass through the network.

        Args:
            x: Input batch [batch_size, input_dim]
            training: Whether in training mode (unused in this simple model)

        Returns:
            Logits [batch_size, output_dim]
        """
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


class StatefulMLP(nnx.Module):
    """Multi-Layer Perceptron with Batch Normalization for stateful training.

    This model demonstrates how to handle stateful components (like batch
    normalization) in distributed multi-step training. The batch normalization
    layers maintain running statistics that are updated during training.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs
    ):
        """Initialize the stateful MLP with batch normalization.

        Args:
            input_dim: Size of input features
            hidden_dim: Size of hidden layer
            output_dim: Size of output layer (number of classes)
            rngs: Random number generators for initialization
        """
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        # Explicitly set dtype=jnp.float32 for batch norm to prevent type promotion
        self.bn1 = nnx.BatchNorm(hidden_dim, dtype=jnp.float32, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.bn2 = nnx.BatchNorm(hidden_dim, dtype=jnp.float32, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, training: bool = True) -> jax.Array:
        """Forward pass through the network with batch normalization.

        Args:
            x: Input batch [batch_size, input_dim]
            training: Whether in training mode (affects batch norm behavior)

        Returns:
            Logits [batch_size, output_dim]
        """
        x = self.linear1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nnx.relu(x)

        x = self.linear2(x)
        x = self.bn2(x, use_running_average=not training)
        x = nnx.relu(x)

        x = self.linear3(x)
        return x


def create_binary_classification_dataset(
    key: jax.Array, num_samples: int = 128, num_features: int = 8
):
    """Create a synthetic binary classification dataset.

    Args:
        key: Random key for data generation
        num_samples: Number of training samples
        num_features: Number of input features

    Returns:
        Tuple of (features, labels) where labels are 0 or 1
    """
    # Generate random features
    features = jax.random.normal(key, (num_samples, num_features))

    # Create separable classes based on feature combinations
    # Class 1 if sum of first half features > sum of second half features
    half = num_features // 2
    class_score = jnp.sum(features[:, :half], axis=1) - jnp.sum(
        features[:, half:], axis=1
    )
    labels = (class_score > 0).astype(jnp.float32)

    return features, labels


def plot_training_progress(
    loss_history, title="Training Progress", device_name="Device"
):
    """Visualize training progress with ASCII art loss curve.

    Args:
        loss_history: Array of loss values over training steps
        title: Title for the plot
        device_name: Name of the device used for training
    """
    if loss_history is None or len(loss_history) == 0:
        print(f"No training data to visualize for {device_name}")
        return

    print(f"\n=== {title} on {device_name} ===")
    print(f"Training steps: {len(loss_history)}")
    print(f"Initial loss: {float(loss_history[0]):.6f}")
    print(f"Final loss: {float(loss_history[-1]):.6f}")

    # Calculate training metrics
    loss_reduction = float(loss_history[0] - loss_history[-1])
    reduction_percent = (loss_reduction / float(loss_history[0])) * 100
    print(f"Loss reduction: {loss_reduction:.6f} ({reduction_percent:.1f}%)")

    # ASCII visualization
    if len(loss_history) > 1:
        print("\nLoss curve (● = loss value, · = baseline):")

        # Normalize for display
        min_loss = float(jnp.min(loss_history))
        max_loss = float(jnp.max(loss_history))
        loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

        # Show progression
        for i in range(0, len(loss_history), max(1, len(loss_history) // 15)):
            loss_val = float(loss_history[i])
            normalized = (loss_val - min_loss) / loss_range
            bar_length = int(normalized * 50)  # 50 character width
            bar = "●" * bar_length + "·" * (50 - bar_length)
            print(f"Step {i:3d}: {bar} {loss_val:.6f}")

    # Training assessment
    if reduction_percent > 10:
        print(f"✓ {device_name}: Strong convergence - model learning effectively")
    elif reduction_percent > 3:
        print(f"✓ {device_name}: Moderate convergence - model is learning")
    else:
        print(f"⚠ {device_name}: Slow convergence - may need tuning")


# Cluster configuration for multi-party computation
cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        {"name": "node_2", "endpoint": "127.0.0.1:61922"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1", "node_2"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        "TEE0": {"kind": "TEE", "members": ["node_2"], "config": {}},
    },
})


@mp.function
def multi_step_training_ppu(
    model_graphdef: nnx.GraphDef,
    initial_model_state: nnx.State,
    initial_opt_state,
    train_features: jax.Array,
    train_labels: jax.Array,
    num_steps: int = 20,
):
    """Multi-step training on PPU with stateful model and loss tracking.

    This function demonstrates a complete training loop:
    1. Initialize training state
    2. Loop over multiple training steps
    3. Compute gradients and update parameters
    4. Update model state (including batch norm statistics)
    5. Track loss progression

    Args:
        model_graphdef: NNX model graph definition
        initial_model_state: Initial model parameters and state
        initial_opt_state: Initial optimizer state
        train_features: Training input features
        train_labels: Training labels
        num_steps: Number of training iterations

    Returns:
        Tuple of (loss_history, final_model_state, final_opt_state)
    """

    def single_training_step(model_state, opt_state, x, y):
        """Perform one training step with gradient computation and updates."""

        def loss_function(state):
            # Recreate model from state
            model = nnx.merge(model_graphdef, state)

            # Forward pass in training mode (updates batch norm stats)
            logits = model(x, training=True)

            # Cross-entropy loss for binary classification
            targets = jax.nn.one_hot(y.astype(jnp.int32), 2)
            loss = optax.softmax_cross_entropy(logits, targets).mean()

            return loss

        # Compute loss and gradients
        loss, gradients = jax.value_and_grad(loss_function)(model_state)

        # Extract parameter gradients (excluding batch norm statistics)
        param_grads = nnx.state(gradients, nnx.Param)

        # Update parameters using optimizer
        optimizer = optax.adam(learning_rate=0.01)
        updates, new_opt_state = optimizer.update(param_grads, opt_state)
        new_model_state = optax.apply_updates(model_state, updates)

        return loss, new_model_state, new_opt_state

    def training_loop(initial_state, initial_opt_state, features, labels, steps):
        """Execute the complete multi-step training loop."""

        def loop_body(step, carry):
            current_state, current_opt_state, loss_array = carry

            # Perform one training step
            step_loss, new_state, new_opt_state = single_training_step(
                current_state, current_opt_state, features, labels
            )

            # Record loss for this step
            updated_losses = loss_array.at[step].set(step_loss)

            return new_state, new_opt_state, updated_losses

        # Initialize loss tracking
        loss_history = jnp.zeros(steps)

        # Execute training loop
        final_state, final_opt_state, final_losses = jax.lax.fori_loop(
            0, steps, loop_body, (initial_state, initial_opt_state, loss_history)
        )

        return final_losses, final_state, final_opt_state

    # Generate training data on PPU
    features = mp.device("P0")(lambda: train_features)()
    labels = mp.device("P0")(lambda: train_labels)()

    # Run multi-step training on PPU
    result = mp.device("P0", fe_type="nnx")(training_loop)(
        initial_model_state, initial_opt_state, features, labels, num_steps
    )

    return result


@mp.function
def stateful_multi_step_training_ppu(
    model_graphdef: nnx.GraphDef,
    initial_model_state: nnx.State,
    initial_opt_state,
    train_features: jax.Array,
    train_labels: jax.Array,
    num_steps: int = 20,
):
    """Multi-step training on PPU with stateful model (batch normalization).

    This function demonstrates training with stateful components where
    the model maintains running statistics that need to be properly
    updated during training. Key differences from stateless training:

    1. Model state includes both parameters and batch norm statistics
    2. Forward pass updates running mean/variance in training mode
    3. State updates must preserve both gradients and statistics
    4. Proper separation of trainable vs non-trainable state

    Args:
        model_graphdef: NNX model graph definition (with batch norm)
        initial_model_state: Initial model parameters and batch norm state
        initial_opt_state: Initial optimizer state
        train_features: Training input features
        train_labels: Training labels
        num_steps: Number of training iterations

    Returns:
        Tuple of (loss_history, final_model_state, final_opt_state)
    """

    def single_training_step_stateful(model_state, opt_state, x, y):
        """Perform one training step with proper state management."""

        def loss_function(state):
            # Recreate model from current state
            model = nnx.merge(model_graphdef, state)

            # Forward pass in training mode (updates batch norm running stats)
            logits = model(x, training=True)

            # Cross-entropy loss for binary classification
            targets = jax.nn.one_hot(y.astype(jnp.int32), 2)
            loss = optax.softmax_cross_entropy(logits, targets).mean()

            # Return both loss and updated model state (with batch norm updates)
            updated_state = nnx.state(model)
            return loss, updated_state

        # Compute loss and gradients, capturing updated batch norm stats
        (loss, updated_state), gradients = jax.value_and_grad(
            loss_function, has_aux=True
        )(model_state)

        # Split gradients: trainable parameters vs batch norm statistics
        param_grads = nnx.state(gradients, nnx.Param)  # Only trainable params

        # Update trainable parameters using optimizer
        optimizer = optax.adam(learning_rate=0.01)
        updates, new_opt_state = optimizer.update(param_grads, opt_state)

        # Apply updates only to parameters (not batch norm stats)
        param_state = nnx.state(updated_state, nnx.Param)
        updated_params = optax.apply_updates(param_state, updates)

        # Get updated batch norm statistics from the forward pass
        updated_bn_stats = nnx.state(updated_state, nnx.BatchStat)

        # Ensure batch norm statistics remain float32 (prevent dtype drift)
        updated_bn_stats = jax.tree.map(
            lambda x: (
                x.astype(jnp.float32)
                if hasattr(x, "dtype") and x.dtype != jnp.float32
                else x
            ),
            updated_bn_stats,
        )

        # Combine updated parameters with updated batch norm stats
        new_model_state = nnx.State.merge(updated_params, updated_bn_stats)

        return loss, new_model_state, new_opt_state

    def stateful_training_loop(
        initial_state, initial_opt_state, features, labels, steps
    ):
        """Execute multi-step training loop preserving model state."""

        def loop_body(step, carry):
            current_state, current_opt_state, loss_array = carry

            # Perform one training step
            step_loss, new_state, new_opt_state = single_training_step_stateful(
                current_state, current_opt_state, features, labels
            )

            # Record loss for this step
            updated_losses = loss_array.at[step].set(step_loss)

            return new_state, new_opt_state, updated_losses

        # Initialize tracking arrays
        loss_history = jnp.zeros(steps)

        # Execute training loop (same as stateless version)
        final_state, final_opt_state, final_losses = jax.lax.fori_loop(
            0,
            steps,
            loop_body,
            (initial_state, initial_opt_state, loss_history),
        )

        return final_losses, final_state, final_opt_state

    # Generate training data on PPU
    features = mp.device("P0")(lambda: train_features)()
    labels = mp.device("P0")(lambda: train_labels)()

    # Run stateful multi-step training on PPU
    training_result = mp.device("P0", fe_type="nnx")(stateful_training_loop)(
        initial_model_state, initial_opt_state, features, labels, num_steps
    )

    # Return the result (contains loss history and batch norm statistics)
    # Note: Model and optimizer states cannot be serialized through MPLang distributed execution
    return training_result


def main():
    """Main training demonstration with both stateless and stateful models."""
    print("=" * 80)
    print("Multi-Step Training: Stateless vs Stateful NNX Models")
    print("=" * 80)

    # 1. Create the dataset
    print("\n1. Preparing Dataset")
    print("-" * 40)

    dataset_key = jax.random.PRNGKey(42)
    train_x, train_y = create_binary_classification_dataset(
        dataset_key, num_samples=128, num_features=8
    )

    print(f"Dataset: {train_x.shape[0]} samples, {train_x.shape[1]} features")
    print(f"Classes: {jnp.unique(train_y.astype(jnp.int32))}")
    print(f"Class distribution: {jnp.bincount(train_y.astype(jnp.int32))}")

    # 2. Initialize simple model (stateless)
    print("\n2. Initializing Simple MLP (Stateless)")
    print("-" * 40)

    model_key = jax.random.PRNGKey(123)
    simple_model = MultiStepMLP(
        input_dim=8, hidden_dim=16, output_dim=2, rngs=nnx.Rngs(model_key)
    )

    # Split model for functional API
    simple_graphdef, simple_initial_state = nnx.split(simple_model)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate=0.01)
    simple_opt_state = optimizer.init(nnx.state(simple_model, nnx.Param))

    # Count parameters
    param_count = sum(
        x.size for x in jax.tree.leaves(nnx.state(simple_model, nnx.Param))
    )
    print("Model architecture: [8 → 16 (ReLU) → 2]")
    print(f"Total parameters: {param_count}")
    print("Focus: Simple multi-step training without state")

    # 3. Initialize stateful model (with batch normalization)
    print("\n3. Initializing Stateful MLP (with Batch Normalization)")
    print("-" * 40)

    stateful_key = jax.random.PRNGKey(456)
    stateful_model = StatefulMLP(
        input_dim=8, hidden_dim=16, output_dim=2, rngs=nnx.Rngs(stateful_key)
    )

    # Split stateful model for functional API
    stateful_graphdef, stateful_initial_state = nnx.split(stateful_model)

    # Initialize optimizer for stateful model
    stateful_opt_state = optimizer.init(nnx.state(stateful_model, nnx.Param))

    # Count parameters (including batch norm)
    stateful_param_count = sum(
        x.size for x in jax.tree.leaves(nnx.state(stateful_model, nnx.Param))
    )
    bn_stat_count = sum(
        x.size for x in jax.tree.leaves(nnx.state(stateful_model, nnx.BatchStat))
    )
    print("Model architecture: [8 → 16 (BN+ReLU) → 16 (BN+ReLU) → 2]")
    print(f"Trainable parameters: {stateful_param_count}")
    print(f"Batch norm statistics: {bn_stat_count}")
    print("Focus: Stateful training with running statistics")

    # 4. Setup training environment
    print("\n4. Setting Up Training Environment")
    print("-" * 40)

    # Initialize simulator
    simulator = mp.Simulator(cluster_spec)
    print("Device configured: PPU (Plaintext Processing Unit)")
    print("Training will demonstrate multi-step learning on PPU")
    print("Data owner: node_0 (will fetch results from this party)")

    num_training_steps = 25

    # 5. Multi-step training - Simple Model
    print("\n5. Multi-Step Training - Simple Model (Stateless)")
    print("-" * 40)

    simple_result = mp.evaluate(
        simulator,
        multi_step_training_ppu,
        simple_graphdef,
        simple_initial_state,
        simple_opt_state,
        train_x,
        train_y,
        num_training_steps,
    )
    raw_result = mp.fetch(simulator, simple_result)[0]
    final_losses, _, _ = raw_result
    plot_training_progress(final_losses, "Simple Model Training", "PPU")

    # 6. Multi-step training - Stateful Model
    print("\n6. Multi-Step Training - Stateful Model (with Batch Norm)")
    print("-" * 40)

    stateful_result = mp.evaluate(
        simulator,
        stateful_multi_step_training_ppu,
        stateful_graphdef,
        stateful_initial_state,
        stateful_opt_state,
        train_x,
        train_y,
        num_training_steps,
    )
    raw_result_stateful = mp.fetch(simulator, stateful_result)[0]
    final_losses_stateful, _, _ = raw_result_stateful

    plot_training_progress(final_losses_stateful, "Stateful Model Training", "PPU")

    # 7. Training Comparison and Analysis
    print("\n7. Training Comparison and Analysis")
    print("-" * 40)

    simple_loss = final_losses
    stateful_loss = final_losses_stateful

    # Calculate improvements
    simple_improvement = (
        (float(simple_loss[0]) - float(simple_loss[-1])) / float(simple_loss[0]) * 100
    )
    stateful_improvement = (
        (float(stateful_loss[0]) - float(stateful_loss[-1]))
        / float(stateful_loss[0])
        * 100
    )

    print("Simple Model:")
    print(f"  Final loss: {float(simple_loss[-1]):.6f}")
    print(f"  Improvement: {simple_improvement:.1f}%")

    print("Stateful Model:")
    print(f"  Final loss: {float(stateful_loss[-1]):.6f}")
    print(f"  Improvement: {stateful_improvement:.1f}%")

    if stateful_improvement > simple_improvement:
        print("✓ Stateful model (with batch norm) achieved better convergence")
    else:
        print("✓ Simple model converged effectively without added complexity")

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("• Stateless models: Simple, predictable, fewer parameters to manage")
    print("• Stateful models: Batch norm improves training stability and convergence")
    print("• State management: Proper separation of parameters vs statistics")
    print("• NNX Functional API: Seamless integration with MPLang computation")
    print("• Multi-step training: Efficient loops with jax.lax.fori_loop")
    print("• Distributed training: Both models work identically on PPU")
    print("• Loss tracking: Real-time monitoring of training progress")
    print("• MPLang limitations: Complex state tracking requires careful design")
    print(
        "• Focus on core functionality: Loss convergence demonstrates model effectiveness"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

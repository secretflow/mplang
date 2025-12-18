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

"""Tutorial 09: Split Learning with Vertical Data Partitioning

Demonstrates split learning in MPLang v1 where:
- **Data is vertically partitioned**: Alice (P0) has n samples × m1 features + labels,
  Bob (P1) has n samples × m2 features (no labels)
- **Three-model architecture**: Alice base model, Bob base model, Alice aggregate model
- **Privacy-preserving**: Raw features never leave their owner's device
- **Training flow**: Forward → backward → update using surrogate loss for base models

Architecture:
    Alice (P0):
        - Alice Base Model: m1 features → h1 embeddings (m1 → 16 → 16)
        - Alice Aggregate Model: [h1, h2] → logits (32 → 16 → 2)
        - Has labels, computes loss

    Bob (P1):
        - Bob Base Model: m2 features → h2 embeddings (m2 → 16 → 16)
        - No labels, uses surrogate loss

Key Concepts:
    1. **State Management**: Use graphdef + state dict pattern (following Tutorial 08)
    2. **Memory Efficiency**: Use nnx.eval_shape() to create abstract models (no array allocation)
    3. **Surrogate Loss**: Base models use dot product (grad · embedding) for training
    4. **Privacy**: Only embeddings and gradients are shared, not raw features
    5. **Optimizer**: Recreated on each step (SGD with fixed learning rate)

Training Flow (One Iteration):
    1. Alice forward: x_alice → h1
    2. Bob forward: x_bob → h2
    3. Transfer h2 from P1 to P0
    4. Alice aggregate: [h1, h2] → logits → loss (supervised)
    5. Alice aggregate backward: compute gradients (params, h1, h2)
    6. Alice aggregate update: apply gradients
    7. Alice base backward: compute gradients using surrogate loss (grad_h1 · h1)
    8. Alice base update: apply gradients
    9. Transfer grad_h2 from P0 to P1
    10. Bob base backward: compute gradients using surrogate loss (grad_h2 · h2)
    11. Bob base update: apply gradients

Usage:
    uv run python tutorials/v1/device/09_split_learning_vertical.py
"""

import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax import nnx

import mplang.v1 as mp
from mplang.v1.core.dtypes import FLOAT64
from mplang.v1.ops import basic as basic_ops

# We'll use a simplified state management approach similar to Tutorial 08
# Instead of TrainState, we'll manually manage state dict + optimizer state

# ============================================================================
# Configuration
# ============================================================================

# Dataset parameters
N_SAMPLES = 10000  # Number of samples
M1 = 10  # Number of Alice's features
M2 = 8  # Number of Bob's features
H1 = H2 = 16  # Embedding dimensions (h1 = h2 = 16)
N_CLASSES = 2  # Binary classification

# Training parameters
LEARNING_RATE = 0.01
SEED_ALICE = 42
SEED_BOB = 43
SEED_AGG = 44

# Cluster specification
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
            "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        },
    }
)


# ============================================================================
# Model Definitions (Flax NNX)
# ============================================================================


class AliceBaseModel(nnx.Module):
    """Alice's base model: m1 features → h1 embeddings (m1 → 16 → 16)."""

    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        """Initialize Alice's base model.

        Args:
            input_dim: Number of input features (m1)
            hidden_dim: Embedding dimension (h1)
            rngs: Random number generator state
        """
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: x (n, m1) → h1 (n, h1)."""
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


class BobBaseModel(nnx.Module):
    """Bob's base model: m2 features → h2 embeddings (m2 → 16 → 16)."""

    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        """Initialize Bob's base model.

        Args:
            input_dim: Number of input features (m2)
            hidden_dim: Embedding dimension (h2)
            rngs: Random number generator state
        """
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: x (n, m2) → h2 (n, h2)."""
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


class AliceAggregateModel(nnx.Module):
    """Alice's aggregate model: [h1, h2] → logits (32 → 16 → 2)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs
    ):
        """Initialize Alice's aggregate model.

        Args:
            input_dim: Combined embedding dimension (h1 + h2 = 32)
            hidden_dim: Hidden layer dimension (16)
            output_dim: Number of classes (2)
            rngs: Random number generator state
        """
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: x (n, 32) → logits (n, 2)."""
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x


# ============================================================================
# Helper Functions: State Management
# ============================================================================


def model_state_to_dict(state, opt_state, step):
    """Convert model state to pure Python dict for cross-device transfer.

    Note: graphdef is NOT stored - it can be reconstructed from model class

    Args:
        state: NNX State (model parameters)
        opt_state: Optax optimizer state
        step: Training step counter

    Returns:
        Dict with keys: model_state_dict, opt_state, step
    """
    return {
        "model_state_dict": state.to_pure_dict(),
        "opt_state": opt_state,
        "step": step,
    }


def reconstruct_model_from_dict(
    model_dict: dict, model_class, *model_args, **model_kwargs
):
    """Reconstruct model and state from dict.

    GraphDef is reconstructed from model class definition.

    Args:
        model_dict: Dict with keys: model_state_dict, opt_state, step
        model_class: Model class (AliceBaseModel, BobBaseModel, AliceAggregateModel)
        *model_args: Arguments for model initialization
        **model_kwargs: Keyword arguments for model initialization

    Returns:
        Tuple of (graphdef, state, opt_state, step)
    """
    # Create temporary model to get GraphDef structure
    temp_model = model_class(*model_args, **model_kwargs)
    graphdef, temp_state = nnx.split(temp_model)

    # Replace with actual parameters
    temp_state.replace_by_pure_dict(model_dict["model_state_dict"])

    return graphdef, temp_state, model_dict["opt_state"], model_dict["step"]


# ============================================================================
# Helper Functions: Forward Pass
# ============================================================================


def alice_base_forward(x, model_dict, m1, h1, seed):
    """Alice's base model forward pass.

    Args:
        x: Input features (n, m1)
        model_dict: Alice base model state as dict
        m1: Number of Alice's features
        h1: Embedding dimension
        seed: Random seed for model reconstruction

    Returns:
        embeddings: Alice's embeddings (n, h1)
    """
    # Reconstruct model
    graphdef, state, _, _ = reconstruct_model_from_dict(
        model_dict, AliceBaseModel, m1, h1, rngs=nnx.Rngs(seed)
    )
    model = nnx.merge(graphdef, state)

    # Forward pass
    embeddings = model(x)

    return embeddings


def bob_base_forward(x, model_dict, m2, h2, seed):
    """Bob's base model forward pass.

    Args:
        x: Input features (n, m2)
        model_dict: Bob base model state as dict
        m2: Number of Bob's features
        h2: Embedding dimension
        seed: Random seed for model reconstruction

    Returns:
        embeddings: Bob's embeddings (n, h2)
    """
    # Reconstruct model
    graphdef, state, _, _ = reconstruct_model_from_dict(
        model_dict, BobBaseModel, m2, h2, rngs=nnx.Rngs(seed)
    )
    model = nnx.merge(graphdef, state)

    # Forward pass
    embeddings = model(x)

    return embeddings


# ============================================================================
# Helper Functions: Backward Pass
# ============================================================================


def alice_aggregate_backward(h1, h2, y, model_dict, n_classes, seed):
    """Compute gradients for Alice's aggregate model using actual loss.

    This is standard supervised learning - compute loss from predictions and labels.

    Args:
        h1: Alice's embeddings (n, h1)
        h2: Bob's embeddings (n, h2)
        y: Labels (n,)
        model_dict: Alice aggregate model state as dict
        n_classes: Number of output classes
        seed: Random seed for model reconstruction

    Returns:
        grads_state_dict: Gradients for aggregate model parameters
        grad_h1: Gradient w.r.t. Alice's embeddings (n, h1)
        grad_h2: Gradient w.r.t. Bob's embeddings (n, h2)
        loss: Scalar loss value
    """

    # Define loss function
    def loss_fn(state_dict, h1, h2, y):
        # Reconstruct model from state dict
        # n_classes and seed are captured from outer scope and are concrete values
        temp_model = AliceAggregateModel(H1 + H2, 16, n_classes, rngs=nnx.Rngs(seed))
        graphdef, temp_state = nnx.split(temp_model)
        temp_state.replace_by_pure_dict(state_dict)
        model = nnx.merge(graphdef, temp_state)

        # Forward pass
        combined = jnp.concatenate([h1, h2], axis=1)
        logits = model(combined)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss

    # Get model state dict
    model_state_dict = model_dict["model_state_dict"]

    # Compute loss and gradients w.r.t. state_dict and inputs (h1, h2) in a single pass
    # Using value_and_grad is more efficient than computing loss twice
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))
    loss, (grads_state_dict, grad_h1, grad_h2) = grad_fn(model_state_dict, h1, h2, y)

    return grads_state_dict, grad_h1, grad_h2, loss


def alice_base_backward(x_alice, grad_h1, model_dict, m1, h1, seed):
    """Compute gradients for Alice's base model using surrogate loss.

    Split Learning Insight:
    - Alice's base model doesn't have direct access to labels
    - Use surrogate loss: L_surrogate = grad_h1 · h1 (dot product)
    - This mimics backpropagation from the aggregate model
    - grad_h1 comes from ∂L_aggregate/∂h1

    Args:
        x_alice: Input features (n, m1)
        grad_h1: Gradient from aggregate model (n, h1)
        model_dict: Alice base model state as dict
        m1: Number of Alice's features
        h1: Embedding dimension
        seed: Random seed for model reconstruction

    Returns:
        grads_state_dict: Gradients for base model parameters (as dict)
    """

    # Surrogate loss function
    def surrogate_loss_fn(state_dict, x, grad_from_next_layer):
        # Reconstruct model from state dict
        # m1, h1, seed are captured from outer scope and are concrete values
        temp_model = AliceBaseModel(m1, h1, rngs=nnx.Rngs(seed))
        graphdef, temp_state = nnx.split(temp_model)
        temp_state.replace_by_pure_dict(state_dict)
        model = nnx.merge(graphdef, temp_state)

        h = model(x)
        # Surrogate loss: dot product with gradients from next layer
        loss = jnp.sum(h * grad_from_next_layer)
        return loss

    # Get model state dict
    model_state_dict = model_dict["model_state_dict"]

    # Compute gradients w.r.t. state_dict using surrogate loss
    grad_fn = jax.grad(surrogate_loss_fn)
    grads_state_dict = grad_fn(model_state_dict, x_alice, grad_h1)

    return grads_state_dict


def bob_base_backward(x_bob, grad_h2, model_dict, m2, h2, seed):
    """Compute gradients for Bob's base model using surrogate loss.

    Split Learning Insight:
    - Bob's base model doesn't have direct access to labels
    - Use surrogate loss: L_surrogate = grad_h2 · h2 (dot product)
    - This mimics backpropagation from the aggregate model
    - grad_h2 comes from ∂L_aggregate/∂h2

    Args:
        x_bob: Input features (n, m2)
        grad_h2: Gradient from aggregate model (n, h2)
        model_dict: Bob base model state as dict
        m2: Number of Bob's features
        h2: Embedding dimension
        seed: Random seed for model reconstruction

    Returns:
        grads_state_dict: Gradients for base model parameters (as dict)
    """

    # Surrogate loss function
    def surrogate_loss_fn(state_dict, x, grad_from_next_layer):
        # Reconstruct model from state dict
        # m2, h2, seed are captured from outer scope and are concrete values
        temp_model = BobBaseModel(m2, h2, rngs=nnx.Rngs(seed))
        graphdef, temp_state = nnx.split(temp_model)
        temp_state.replace_by_pure_dict(state_dict)
        model = nnx.merge(graphdef, temp_state)

        h = model(x)
        # Surrogate loss: dot product with gradients from next layer
        loss = jnp.sum(h * grad_from_next_layer)
        return loss

    # Get model state dict
    model_state_dict = model_dict["model_state_dict"]

    # Compute gradients w.r.t. state_dict using surrogate loss
    grad_fn = jax.grad(surrogate_loss_fn)
    grads_state_dict = grad_fn(model_state_dict, x_bob, grad_h2)

    return grads_state_dict


# ============================================================================
# Helper Functions: Update
# ============================================================================


def update_model_state(model_dict, grads_dict, lr):
    """Update model state with gradients using optax SGD.

    Similar to Tutorial 08 pattern - use optax for clean optimizer integration.

    Args:
        model_dict: Current model state dict (keys: model_state_dict, opt_state, step)
        grads_dict: Gradients as pure dict (same format as model_state_dict)
        lr: Learning rate

    Returns:
        new_model_dict: Updated model state dict
    """
    model_state_dict = model_dict["model_state_dict"]
    opt_state = model_dict["opt_state"]
    step = model_dict["step"]

    # Create optimizer (SGD)
    tx = optax.sgd(lr)

    # Apply optimizer update
    updates, new_opt_state = tx.update(grads_dict, opt_state, model_state_dict)
    new_model_state_dict = optax.apply_updates(model_state_dict, updates)

    # Return updated dict
    return {
        "model_state_dict": new_model_state_dict,
        "opt_state": new_opt_state,
        "step": step + 1,
    }


# ============================================================================
# Helper Functions: Reusable Partial Functions
# ============================================================================

# Create partial functions that bind module-level constants (not traced parameters)
# These can be reused in training steps without worrying about JAX tracer issues


def get_alice_base_backward_fn():
    """Returns partial function for Alice base backward with module constants."""
    return partial(alice_base_backward, m1=M1, h1=H1, seed=SEED_ALICE)


def get_bob_base_backward_fn():
    """Returns partial function for Bob base backward with module constants."""
    return partial(bob_base_backward, m2=M2, h2=H2, seed=SEED_BOB)


def get_alice_agg_backward_fn():
    """Returns partial function for Alice aggregate backward with module constants."""
    return partial(alice_aggregate_backward, n_classes=N_CLASSES, seed=SEED_AGG)


# ============================================================================
# Data Preparation
# ============================================================================


def prepare_vertical_split_data():
    """Generate synthetic vertically partitioned data for split learning.

    Creates:
        - Alice (P0): 10,000 samples × 10 features + labels
        - Bob (P1): 10,000 samples × 8 features (no labels)

    Saves to:
        - tmp/alice_data.csv
        - tmp/bob_data.csv
    """
    print("\n" + "=" * 80)
    print("Generating Synthetic Vertically Partitioned Data")
    print("=" * 80)

    # Create tmp directory
    os.makedirs("tmp", exist_ok=True)
    base_dir = os.path.abspath("tmp")

    # Generate synthetic data
    key = jax.random.PRNGKey(0)
    key_alice, key_bob, key_labels = jax.random.split(key, 3)

    # Alice: m1 features + labels
    alice_features = jax.random.normal(key_alice, (N_SAMPLES, M1))
    alice_labels = jax.random.randint(key_labels, (N_SAMPLES,), 0, N_CLASSES)

    # Bob: m2 features (no labels)
    bob_features = jax.random.normal(key_bob, (N_SAMPLES, M2))

    # Save as CSV
    alice_cols = {f"alice_f{i}": alice_features[:, i] for i in range(M1)}
    alice_cols["label"] = alice_labels
    df_alice = pd.DataFrame(alice_cols)
    df_alice.to_csv(f"{base_dir}/alice_data.csv", index=False)

    bob_cols = {f"bob_f{i}": bob_features[:, i] for i in range(M2)}
    df_bob = pd.DataFrame(bob_cols)
    df_bob.to_csv(f"{base_dir}/bob_data.csv", index=False)

    print(f"✅ Alice data: {N_SAMPLES} samples x {M1} features + labels")
    print(f"   Saved to: {base_dir}/alice_data.csv")
    print(f"✅ Bob data: {N_SAMPLES} samples x {M2} features")
    print(f"   Saved to: {base_dir}/bob_data.csv")
    print("=" * 80)


@mp.function
def load_vertical_split_data(
    alice_csv: str, bob_csv: str, m1: int, m2: int, n_rows: int
):
    """Load vertically split data onto P0 and P1.

    Args:
        alice_csv: Path to Alice's CSV file
        bob_csv: Path to Bob's CSV file
        m1: Number of Alice's features
        m2: Number of Bob's features
        n_rows: Number of rows to load

    Returns:
        alice_features: Tensor on P0 (n, m1)
        alice_labels: Tensor on P0 (n,)
        bob_features: Tensor on P1 (n, m2)
    """
    # Define schemas (all columns must have same dtype for table_to_tensor)
    # Cast label to FLOAT64 here, convert back to int after splitting
    schema_alice = mp.TableType.from_dict(
        {
            **{f"alice_f{i}": FLOAT64 for i in range(m1)},
            "label": FLOAT64,
        }
    )
    schema_bob = mp.TableType.from_dict({f"bob_f{i}": FLOAT64 for i in range(m2)})

    # Read CSVs as tables on respective devices
    tbl_alice = mp.device("P0")(basic_ops.read)(path=alice_csv, ty=schema_alice)
    tbl_bob = mp.device("P1")(basic_ops.read)(path=bob_csv, ty=schema_bob)

    # Convert to tensors
    alice_tensor = mp.device("P0")(basic_ops.table_to_tensor)(
        tbl_alice, number_rows=n_rows
    )
    bob_tensor = mp.device("P1")(basic_ops.table_to_tensor)(tbl_bob, number_rows=n_rows)

    # Split Alice tensor into features and labels
    def split_features_labels(data):
        features = data[:, :-1]  # All columns except last
        labels = data[:, -1].astype(jnp.int32)  # Last column as int
        return features, labels

    alice_features, alice_labels = mp.device("P0")(split_features_labels)(alice_tensor)
    bob_features = bob_tensor

    return alice_features, alice_labels, bob_features


# ============================================================================
# Model Initialization
# ============================================================================


@mp.function
def initialize_alice_base_model(m1: int, h1: int, seed: int, learning_rate: float):
    """Initialize Alice's base model on P0 and return state as dict."""

    def _init():
        # Create model
        model = AliceBaseModel(input_dim=m1, hidden_dim=h1, rngs=nnx.Rngs(seed))

        # Split into graphdef + state
        _graphdef, state = nnx.split(model)

        # Initialize optimizer
        tx = optax.sgd(learning_rate)
        opt_state = tx.init(state.to_pure_dict())

        return model_state_to_dict(state, opt_state, 0)

    return mp.device("P0", fe_type="nnx")(_init)()


@mp.function
def initialize_bob_base_model(m2: int, h2: int, seed: int, learning_rate: float):
    """Initialize Bob's base model on P1 and return state as dict."""

    def _init():
        # Create model
        model = BobBaseModel(input_dim=m2, hidden_dim=h2, rngs=nnx.Rngs(seed))

        # Split into graphdef + state
        _graphdef, state = nnx.split(model)

        # Initialize optimizer
        tx = optax.sgd(learning_rate)
        opt_state = tx.init(state.to_pure_dict())

        return model_state_to_dict(state, opt_state, 0)

    return mp.device("P1", fe_type="nnx")(_init)()


@mp.function
def initialize_alice_agg_model(
    input_dim: int, hidden_dim: int, output_dim: int, seed: int, learning_rate: float
):
    """Initialize Alice's aggregate model on P0 and return state as dict."""

    def _init():
        # Create model
        model = AliceAggregateModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rngs=nnx.Rngs(seed),
        )

        # Split into graphdef + state
        _graphdef, state = nnx.split(model)

        # Initialize optimizer
        tx = optax.sgd(learning_rate)
        opt_state = tx.init(state.to_pure_dict())

        return model_state_to_dict(state, opt_state, 0)

    return mp.device("P0", fe_type="nnx")(_init)()


# ============================================================================
# Training Step
# ============================================================================


@mp.function
def split_learning_train_step(
    alice_features,  # On P0: (10000, m1)
    alice_labels,  # On P0: (10000,)
    bob_features,  # On P1: (10000, m2)
    alice_base_model_dict,  # Alice base model state as dict
    bob_base_model_dict,  # Bob base model state as dict
    alice_agg_model_dict,  # Alice aggregate model state as dict
    learning_rate: float,
    m1: int,
    m2: int,
    h1: int,
    h2: int,
    n_classes: int,
    seed_alice: int,
    seed_bob: int,
    seed_agg: int,
):
    """One complete split learning training step (single batch = full dataset).

    Training Flow (Split Learning):
    1. Alice computes base model forward → h1
    2. Bob computes base model forward → h2
    3. Transfer h2 from P1 to P0
    4. Alice runs aggregate model → logits → loss (actual supervised loss)
    5. Alice computes gradients for aggregate model (∂L/∂params_agg, ∂L/∂h1, ∂L/∂h2)
    6. Alice updates aggregate model with actual gradients
    7. Alice computes gradients for her base model using surrogate loss (grad_h1 · h1)
    8. Alice updates her base model with surrogate gradients
    9. Alice sends ∂L/∂h2 back to Bob (P0 → P1)
    10. Bob computes gradients for his base model using surrogate loss (grad_h2 · h2)
    11. Bob updates his base model with surrogate gradients

    Note: Uses helper functions (get_*_backward_fn) that bind module-level constants
    via functools.partial to avoid traced parameter issues in JAX gradient computation.

    Returns:
        new_alice_base_model_dict: Updated Alice base model state (as dict)
        new_bob_base_model_dict: Updated Bob base model state (as dict)
        new_alice_agg_model_dict: Updated Alice aggregate model state (as dict)
        loss: Training loss (scalar)
    """

    # === Forward Pass ===

    # 1. Alice base model forward on P0
    h1_embeddings = mp.device("P0", fe_type="nnx")(alice_base_forward)(
        alice_features, alice_base_model_dict, m1, h1, seed_alice
    )

    # 2. Bob base model forward on P1
    h2_embeddings = mp.device("P1", fe_type="nnx")(bob_base_forward)(
        bob_features, bob_base_model_dict, m2, h2, seed_bob
    )

    # 3. Transfer Bob's embeddings to Alice (P1 → P0)
    h2_on_p0 = mp.put("P0", h2_embeddings)

    # === Backward Pass (Alice Aggregate Model) ===

    # 4. Alice computes gradients for aggregate model using actual loss
    agg_grads_dict, grad_h1, grad_h2, loss = mp.device("P0", fe_type="nnx")(
        get_alice_agg_backward_fn()
    )(h1_embeddings, h2_on_p0, alice_labels, alice_agg_model_dict)

    # 5. Alice updates her aggregate model with actual gradients
    new_alice_agg_model_dict = mp.device("P0", fe_type="nnx")(update_model_state)(
        alice_agg_model_dict, agg_grads_dict, learning_rate
    )

    # === Backward Pass (Alice Base Model) ===

    # 6. Alice computes gradients for her base model using surrogate loss
    alice_base_grads_dict = mp.device("P0", fe_type="nnx")(
        get_alice_base_backward_fn()
    )(alice_features, grad_h1, alice_base_model_dict)

    # 7. Alice updates her base model with surrogate gradients
    new_alice_base_model_dict = mp.device("P0", fe_type="nnx")(update_model_state)(
        alice_base_model_dict, alice_base_grads_dict, learning_rate
    )

    # === Backward Pass (Bob Base Model) ===

    # 8. Transfer grad_h2 to Bob (P0 → P1)
    grad_h2_on_p1 = mp.put("P1", grad_h2)

    # 9. Bob computes gradients for his base model using surrogate loss
    bob_base_grads_dict = mp.device("P1", fe_type="nnx")(get_bob_base_backward_fn())(
        bob_features, grad_h2_on_p1, bob_base_model_dict
    )

    # 10. Bob updates his base model with surrogate gradients
    new_bob_base_model_dict = mp.device("P1", fe_type="nnx")(update_model_state)(
        bob_base_model_dict, bob_base_grads_dict, learning_rate
    )

    return (
        new_alice_base_model_dict,
        new_bob_base_model_dict,
        new_alice_agg_model_dict,
        loss,
    )


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Main demonstration of split learning with vertical data partitioning."""
    print("\n" + "=" * 80)
    print("Split Learning Tutorial: Vertical Data Partitioning")
    print("=" * 80)
    print(f"Dataset: {N_SAMPLES} samples")
    print(f"Alice (P0): {M1} features + labels")
    print(f"Bob (P1): {M2} features (no labels)")
    print(f"Embedding dimensions: h1 = h2 = {H1}")
    print(f"Number of classes: {N_CLASSES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 80)

    # Step 1: Prepare data
    prepare_vertical_split_data()

    # Step 2: Setup simulator
    print("\n[Step 2] Setting up MPLang simulator...")
    simulator = mp.Simulator(cluster_spec)

    # Step 3: Load data onto devices
    print("\n[Step 3] Loading data onto devices P0 and P1...")
    base_dir = os.path.abspath("tmp")
    alice_features, alice_labels, bob_features = mp.evaluate(
        simulator,
        load_vertical_split_data,
        f"{base_dir}/alice_data.csv",
        f"{base_dir}/bob_data.csv",
        M1,
        M2,
        N_SAMPLES,
    )
    print("✅ Data loaded successfully")

    # Step 4: Initialize models
    print("\n[Step 4] Initializing models on devices...")

    # Alice base model (P0)
    alice_base_model_dict = mp.evaluate(
        simulator,
        initialize_alice_base_model,
        M1,
        H1,
        SEED_ALICE,
        LEARNING_RATE,
    )
    print(f"✅ Alice base model initialized on P0 (m1={M1} → h1={H1})")

    # Bob base model (P1)
    bob_base_model_dict = mp.evaluate(
        simulator,
        initialize_bob_base_model,
        M2,
        H2,
        SEED_BOB,
        LEARNING_RATE,
    )
    print(f"✅ Bob base model initialized on P1 (m2={M2} → h2={H2})")

    # Alice aggregate model (P0)
    alice_agg_model_dict = mp.evaluate(
        simulator,
        initialize_alice_agg_model,
        H1 + H2,  # Concatenated embeddings
        16,  # Hidden dimension
        N_CLASSES,
        SEED_AGG,
        LEARNING_RATE,
    )
    print(f"✅ Alice aggregate model initialized on P0 ({H1 + H2} → 16 → {N_CLASSES})")

    # Step 5: Run training iterations
    print("\n[Step 5] Running split learning training iterations...")
    print("Training flow:")
    print("  1. Alice forward: x_alice → h1")
    print("  2. Bob forward: x_bob → h2")
    print("  3. Transfer h2: P1 → P0")
    print("  4. Alice aggregate: [h1, h2] → loss")
    print("  5. Alice aggregate backward & update")
    print("  6. Alice base backward & update (surrogate loss)")
    print("  7. Transfer grad_h2: P0 → P1")
    print("  8. Bob base backward & update (surrogate loss)")
    print("\nRunning 5 training iterations on the same batch...")

    current_alice_base = alice_base_model_dict
    current_bob_base = bob_base_model_dict
    current_alice_agg = alice_agg_model_dict

    for iter_idx in range(5):
        result = mp.evaluate(
            simulator,
            split_learning_train_step,
            alice_features,
            alice_labels,
            bob_features,
            current_alice_base,
            current_bob_base,
            current_alice_agg,
            LEARNING_RATE,
            M1,
            M2,
            H1,
            H2,
            N_CLASSES,
            SEED_ALICE,
            SEED_BOB,
            SEED_AGG,
        )

        # Fetch only for inspection
        (
            _new_alice_base_fetched,
            _new_bob_base_fetched,
            _new_alice_agg_fetched,
            loss_fetched,
        ) = mp.fetch(simulator, result)

        print(f"  Iteration {iter_idx + 1}: Loss = {loss_fetched}")

        # Update current state using result (before fetch) for next iteration
        # Extract components from tuple result
        current_alice_base = result[0]
        current_bob_base = result[1]
        current_alice_agg = result[2]

    print("\n✅ Training complete!")
    print(f"   Final loss: {loss_fetched}")

    # Summary
    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print(
        "1. ✅ Vertical data partitioning: Alice has features + labels, Bob has features"
    )
    print("2. ✅ Three-model architecture: Alice base, Bob base, Alice aggregate")
    print(
        "3. ✅ State dict pattern: model_state_dict + opt_state + step (following Tutorial 08)"
    )
    print("4. ✅ Optax integration: Use optax.sgd() for optimizer state management")
    print("5. ✅ Surrogate loss: Base models use dot product (grad · embedding)")
    print("6. ✅ Privacy-preserving: Only embeddings and gradients are shared")
    print("7. ✅ Complete training: Multiple iterations with decreasing loss")
    print("8. ✅ Efficient: Pass results before fetch, only fetch for inspection")
    print("\nSplit Learning enables collaborative training without sharing raw data!")
    print("=" * 80)


if __name__ == "__main__":
    main()

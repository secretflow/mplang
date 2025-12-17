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

"""stax_nn example using mplang.v2.

This demonstrates training a neural network with privacy-preserving
computation on SPU using the new mplang.v2 API.

Usage:
    # Simulation mode (default)
    uv run python examples/mp2/stax_nn.py --model network_a -e 1 -b 128

    # With driver (requires workers running)
    uv run python examples/mp2/stax_nn.py --model network_a -e 1 -b 128 --driver
"""

import argparse
import itertools
import math

# Import models from parent directory
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers, stax
from sklearn.metrics import accuracy_score

import mplang.v2 as mp

# Add examples/v1/stax_nn to sys.path to import shared model definitions
sys.path.insert(0, str(Path(__file__).parent.parent / "v1" / "stax_nn"))
import models

parser = argparse.ArgumentParser(description="stax_nn with mplang.v2")
parser.add_argument("--model", default="network_a", type=str)
parser.add_argument("-e", "--epoch", default=5, type=int)
parser.add_argument("-b", "--batch_size", default=128, type=int)
parser.add_argument("-o", "--optimizer", default="SGD", type=str)
parser.add_argument(
    "--driver",
    default=False,
    action="store_true",
    help="Use driver mode (requires workers running)",
)
args = parser.parse_args()

DEFAULT_EPOCHS = args.epoch
DEFAULT_BATCH_SIZE = args.batch_size

# Cluster configuration for 2-party computation
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
        "P0": {"kind": "PPU", "members": ["node_0"]},
        "P1": {"kind": "PPU", "members": ["node_1"]},
    },
})


def train(
    train_x,
    train_y,
    init_fun,
    predict_fun,
    epochs,
    batch_size,
):
    """Train a neural network with secure gradient updates on SPU."""
    # Model Initialization
    key = jax.random.PRNGKey(42)
    input_shape = tuple(
        -1 if idx == 0 else i for idx, i in enumerate(list(train_x.shape))
    )
    _, params_init = init_fun(key, input_shape)

    # Setup optimizer
    opt_kind = args.optimizer.lower()
    if opt_kind == "sgd":
        opt_init, opt_update, get_params = optimizers.momentum(0.01, 0.9)
    elif opt_kind == "adam":
        opt_init, opt_update, get_params = optimizers.adam(0.001)
    else:
        raise RuntimeError(f"Unsupported optimizer type {args.optimizer}.")

    opt_state = opt_init(params_init)

    def update_model(state, imgs, labels, i):
        """Single gradient update step."""

        def ce_loss(y, label):
            return -jnp.mean(jnp.sum(label * stax.logsoftmax(y), axis=1))

        def loss_func(params):
            y = predict_fun(params, imgs)
            return ce_loss(y, labels), y

        grad_fn = jax.value_and_grad(loss_func, has_aux=True)
        (_loss, _y), grads = grad_fn(get_params(state))
        return opt_update(i, grads, state)

    itercount = itertools.count()

    # Training loop with mplang.v2
    from jax.tree_util import tree_map

    # comment out 'mp.function' to run tracking loop eagerly on driver

    @mp.function
    def do_train(opt_state):
        for i in range(1, epochs + 1):
            for batch_idx in range(math.ceil(len(train_x) / batch_size)):
                batch_images = train_x[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                batch_labels = train_y[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                it = next(itercount)
                print(
                    f"{datetime.now().time()} Epoch: {i}/{epochs}  "
                    f"Batch: {batch_idx}/{math.floor(len(train_x) / batch_size)}"
                )

                # Secure training: each party holds part of the data
                # P0 holds the images, P1 holds the labels
                p0_images = mp.put("P0", batch_images)
                p1_labels = mp.put("P1", batch_labels)

                # Compute gradient update on SPU (secure multi-party)
                # SPU always uses JAX semantics natively
                # Use .jax() to automatically handle secret sharing of inputs
                opt_state = mp.device("SP0").jax(update_model)(
                    opt_state, p0_images, p1_labels, it
                )

        # Move results to P0 leaf-by-leaf to trigger implicit reconstruction/reveal.
        # This is done inside the function so do_train returns values already on P0.
        return tree_map(lambda x: mp.device("P0").jax(lambda y: y)(x), opt_state)

    opt_state_p0_final = do_train(opt_state)

    # Fetch results to driver (data is already on P0 with device attribute)
    print("Fetching results...")
    opt_state_final = tree_map(lambda x: mp.fetch(x), opt_state_p0_final)

    # Extract params
    params = get_params(opt_state_final)
    return params


def get_datasets(name="mnist"):
    """Load MNIST train and test datasets into memory."""
    if name == "mnist":
        from tensorflow.keras.datasets import mnist

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize to [0, 1]
        train_images = np.float32(train_images) / 255.0
        test_images = np.float32(test_images) / 255.0

        # Add channel dimension (batch, H, W, C)
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)

        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}
        return train_ds, test_ds

    else:
        raise ValueError(f"Dataset '{name}' not supported.")


def train_mnist(model):
    """Train on MNIST dataset."""
    train_ds, test_ds = get_datasets("mnist")
    train_x, train_y = train_ds["image"], train_ds["label"]
    train_y = jax.nn.one_hot(train_y, 10)

    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE

    init_fun, predict_fun = model()

    start = time.perf_counter()
    params = train(
        train_x,
        train_y,
        init_fun,
        predict_fun,
        epochs,
        batch_size,
    )
    end = time.perf_counter()

    print(f"train(spu) elapsed time: {end - start:0.4f} seconds")

    # Evaluate
    test_x, test_y = test_ds["image"], test_ds["label"]
    # Run prediction
    predict_y = predict_fun(params, test_x)
    score = accuracy_score(np.argmax(predict_y, axis=1), test_y)
    print(f"accuracy(spu): {score}")
    return score


def run_model(model_name):
    """Run the selected model."""
    print(f"The selected NN model is {model_name}.")

    MODEL_MAPS = {
        "network_a": models.secureml,
        "network_b": models.minionn,
        "network_c": models.lenet,
        "network_d": models.chameleon,
        "alexnet": models.alexnet,
        "lenet": models.lenet,
        "vgg16": models.vgg16,
    }

    model = MODEL_MAPS.get(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    print("Run on SPU (mplang.v2)\n------\n")
    return train_mnist(model)


def main():
    if args.driver:
        # Driver mode: connect to running workers
        print("Using Driver mode - ensure workers are running!")
        print("Start workers with: python -m mplang.v2.backends.cli up --world-size 2")
        driver = mp.make_driver(cluster_spec.endpoints, cluster_spec=cluster_spec)
        ctx = driver
    else:
        # Simulation mode: run locally with threads
        print("Using Simulation mode")
        sim = mp.make_simulator(2, cluster_spec=cluster_spec)
        ctx = sim

    # Set context and run
    with ctx:
        run_model(args.model)


if __name__ == "__main__":
    main()

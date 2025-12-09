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

import argparse
import itertools
import math
import time
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import models
import numpy as np
import yaml
from jax.example_libraries import stax
from sklearn.metrics import accuracy_score

import mplang.v1 as mp

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("--model", default="network_a", type=str)
parser.add_argument("-c", "--config", default="examples/v1/conf/3pc.yaml", type=str)
parser.add_argument("-e", "--epoch", default=5, type=int)
parser.add_argument("-b", "--batch_size", default=128, type=int)
parser.add_argument("-o", "--optimizer", default="SGD", type=str)
parser.add_argument("--run_cpu", default=False, action="store_true")
args = parser.parse_args()

# Follows https://arxiv.org/pdf/2107.00501.pdf Appendix C.
DEFAULT_EPOCHS = args.epoch
DEFAULT_BATCH_SIZE = args.batch_size


def train(
    train_x,
    train_y,
    init_fun,
    predict_fun,
    epochs,
    batch_size,
    run_on_spu,
):
    # Model Initialization
    key = jax.random.PRNGKey(42)
    input_shape = tuple(
        -1 if idx == 0 else i for idx, i in enumerate(list(train_x.shape))
    )
    _, params_init = init_fun(key, input_shape)
    opt_kind = args.optimizer.lower()
    if opt_kind == "sgd":
        from jax.example_libraries import optimizers

        opt_init, opt_update, get_params = optimizers.momentum(0.01, 0.9)
    elif opt_kind == "adam":
        from jax.example_libraries import optimizers

        opt_init, opt_update, get_params = optimizers.adam(0.001)
    elif opt_kind == "amsgrad":
        from examples.python.utils import optimizers

        opt_init, opt_update, get_params = optimizers.amsgrad(0.001)
    else:
        raise RuntimeError(f"Unsupported optimizer type {args.optimizer}.")
    opt_state = opt_init(params_init)

    def update_model(state, imgs, labels, i):
        def ce_loss(y, label):
            return -jnp.mean(jnp.sum(label * stax.logsoftmax(y), axis=1))

        def loss_func(params):
            y = predict_fun(params, imgs)
            return ce_loss(y, labels), y

        grad_fn = jax.value_and_grad(loss_func, has_aux=True)
        (_loss, _y), grads = grad_fn(get_params(state))
        return opt_update(i, grads, state)

    def identity(x):
        return x

    update_model_jit = jax.jit(update_model)
    itercount = itertools.count()

    print("Start training...")

    # when 'mpd.function' is used, the function will be compiled, or it will be run eagerly.
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
                    f"{datetime.now().time()} Epoch: {i}/{epochs}  Batch: {batch_idx}/{math.floor(len(train_x) / batch_size)}"
                )
                if run_on_spu:
                    # mpd.function could be added recursively inside another mpd.function.
                    @mp.function
                    def secure_update_model(opt_state):
                        p1_batch_images = mp.device("P0")(identity)(batch_images)
                        p2_batch_labels = mp.device("P1")(identity)(batch_labels)
                        # return mpd.device("TEE0")(update_model)(
                        return mp.device("SP0")(update_model)(
                            opt_state, p1_batch_images, p2_batch_labels, it
                        )

                    opt_state = secure_update_model(opt_state)

                else:
                    opt_state = update_model_jit(
                        opt_state, batch_images, batch_labels, it
                    )
        return opt_state

    opt_state = do_train(opt_state)
    if run_on_spu:
        # FIXME: hosts variable is not 'reconstructed' variable.
        print("opt_state:", opt_state)
        opt_state = mp.fetch(None, opt_state)
    return get_params(opt_state)


def get_datasets(name="mnist"):
    """Load MNIST train and test datasets into memory."""
    if name == "mnist":
        from tensorflow.keras.datasets import mnist

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Data preprocessing, consistent with the original code
        # 1. Normalize to the [0, 1] interval
        train_images = np.float32(train_images) / 255.0
        test_images = np.float32(test_images) / 255.0

        # 2. (Optional but recommended) Add a channel dimension to conform to the conventional model input (batch, H, W, C)
        # The MNIST data loaded by Keras is (60000, 28, 28), which needs to be expanded to (60000, 28, 28, 1)
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)

        # 3. Construct a dictionary with the same output format as the original function
        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}

        return train_ds, test_ds

    elif name == "cifar10":
        from tensorflow.keras.datasets import cifar10

        (train_x, train_y), (test_imgs, test_labels) = cifar10.load_data()
        train_x = np.float32(train_x) / 255.0
        train_y = np.squeeze(train_y)
        test_imgs = np.float32(test_imgs) / 255.0
        test_labels = np.squeeze(test_labels)

        # Note: The return format of cifar10 is different from mnist, keeping it as is here
        return (train_x, train_y), (test_imgs, test_labels)

    else:
        raise ValueError(f"Dataset '{name}' not supported.")


def train_mnist(model, run_on_spu: bool = False):
    train_ds, test_ds = get_datasets("mnist")
    train_x, train_y = train_ds["image"], train_ds["label"]
    train_y = jax.nn.one_hot(train_y, 10)

    # Hyper-parameters
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
        run_on_spu,
    )
    end = time.perf_counter()

    env = "spu" if run_on_spu else "cpu"
    print(f"train({env}) elapsed time: {end - start:0.4f} seconds")
    test_x, test_y = test_ds["image"], test_ds["label"]
    predict_y = predict_fun(params, test_x)
    score = accuracy_score(np.argmax(predict_y, axis=1), test_y)
    print(f"accuracy({env}): {score}")
    return score


def run_model(model_name, run_cpu=True):
    print(f"The selected NN model is {args.model}.")

    MODEL_MAPS = {
        "network_a": models.secureml,
        "network_b": models.minionn,
        "network_c": models.lenet,
        "network_d": models.chameleon,
        "alexnet": models.alexnet,
        "lenet": models.lenet,
        "vgg16": models.vgg16,
    }

    fn = partial(train_mnist, MODEL_MAPS.get(model_name))
    if run_cpu:
        print("Run on CPU\n------\n")
        return fn(run_on_spu=False)

    print("Run on SPU\n------\n")
    return fn(run_on_spu=True)


def main():
    if args.run_cpu:
        run_model(args.model, run_cpu=True)
        return

    with open(args.config) as file:
        conf = yaml.safe_load(file)
    cluster_spec = mp.ClusterSpec.from_dict(conf)
    driver = mp.Driver(cluster_spec, timeout=600)
    mp.set_ctx(driver)

    # with open(args.config) as file:
    #     conf = json.load(file)
    # mpd.init(conf["devices"], conf["nodes"])
    return run_model(args.model, run_cpu=False)


if __name__ == "__main__":
    main()

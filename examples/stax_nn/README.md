# Stax NN Example

This example demonstrates how to use SPU to train neural network models privately for image classification.

These models are widely used for evaluation benchmarks in MPC-enabled literature such as [deep-mpc](https://arxiv.org/abs/2107.00501) and [SeureNN](https://eprint.iacr.org/2018/442.pdf).

1. Launch SPU backend runtime

    ```sh
    uv run python -m mplang.runtime.cli up -c examples/conf/3pc.yaml

    # alternative method
    uv run mplang-cli up -c examples/conf/3pc.yaml
    ```

2. Run `stax_nn` example

    ```sh
    uv run examples/stax_nn/stax_nn.py -e 1 -b 1024
    ```

    **Note**: The parameters `-e 1 -b 1024` are required to run this example. Using the default epoch and batch size will cause the traced AST to become too large, leading to a stack overflow during expression visitor operations. A fundamental fix for this issue requires operating at the IR level instead of the AST level.

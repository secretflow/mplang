# Stax NN Example

This example demonstrates how to use SPU to train neural network models privately for image classification.

These models are widely used for evaluation benchmarks in MPC-enabled literature such as [deep-mpc](https://arxiv.org/abs/2107.00501) and [SeureNN](https://eprint.iacr.org/2018/442.pdf).

1. Launch SPU backend runtime

    ```sh
    uv sync --group examples

    uv run examples/stax_nn/stax_nn.py --action up
    ```

2. Run `stax_nn` example

    ```sh
    uv run examples/stax_nn/stax_nn.py -e 1 -b 1024
    ```

    **Note**: We cannot run this example without `-e 1 -b 1024`, since the default epoch and batch size will cause the traced AST to become too large, leading to stack overflow in subsequent expr visitor operations. To fundamentally solve this issue, we need to operate on IR level (rather than AST level).

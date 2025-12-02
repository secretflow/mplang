# XGBoost Evolution: From Numpy to JAX (GPU)

This document outlines the evolution of XGBoost from its simplest Numpy version to
the JAX (GPU) version. The ultimate goal is to implement Secure Boost (sgboost) using
a compilable JAX version.

## Steps of Evolution

1. **Numpy Version**: The initial implementation of XGBoost using Numpy for basic operations.
2. **JAX Version**: Transitioning from Numpy to JAX to leverage automatic differentiation and GPU acceleration.
3. **JAX (GPU) Version**: Optimizing the JAX implementation to fully utilize GPU capabilities.
4. **Secure Boost (sgboost)**: Implementing Secure Boost using the compilable JAX version for enhanced
security and performance.

By following these steps, we aim to achieve a highly efficient and secure implementation of XGBoost.

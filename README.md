# baifs

baifs is a minimal deep learning framework built from scratch in pure Python. It implements a scalar-level automatic differentiation engine and uses it to build and train neural networks without any external deep learning libraries.

## What it does

The core of baifs is a scalar autograd engine (`ScalarGrad`) that tracks operations and computes gradients via reverse-mode backpropagation. On top of this, `TensorGrad` provides a tensor abstraction that supports elementwise operations, dot products, and reductions over nested lists of scalars. Every forward pass builds a dynamic computation graph, and calling `.backward()` propagates gradients through the entire graph.

The framework supports fully connected layers (`Linear`), 2D convolutions (`Conv2D`), residual blocks (`ResidualBlock`), MSE loss, and SGD with gradient clipping. Two model architectures are included: a multi-layer perceptron (`MLP`) and a small residual network (`MicroResNet`).

## Project structure

`tensor.py` defines `ScalarGrad`, `TensorGrad`, and the `tensor()` and `scalar()` factory functions that serve as the user-facing API for constructing differentiable values.

`layers.py` defines `Module`, `Linear`, `Conv2D`, `ResidualBlock`, and helper functions for padding and flattening.

`models.py` defines `MLP` and `MicroResNet` using the layer primitives.

`losses.py` implements mean squared error loss.

`optim.py` implements SGD with gradient clipping.

`solver.py` implements the training loop and validation.

`data.py` provides toy datasets for both MLP and CNN experiments.

`infer.py` provides a simple inference function.

`main.py` is the entry point with `run_mlp()` and `run_cnn()` functions.

## Usage

```python
python main.py
```

## Purpose

baifs is a learning and research scaffold. It is intentionally simple — every operation is traceable to a scalar multiply or add — making it ideal for understanding how autograd, backpropagation, and gradient-based optimization work at the lowest level.
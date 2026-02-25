# __init__.py
from .tensor import ScalarGrad, TensorGrad, scalar, tensor
from .layers import Linear, Conv2D, ResidualBlock, flatten
from .models import MLP, MicroResNet
from .losses import mse_loss
from .optim import sgd
from .solver import train, validate
from .infer import test
from .data import mlp_dataset, cnn_dataset, mlp_test_dataset, cnn_test_dataset

__all__ = [
    "ScalarGrad", "TensorGrad", "scalar", "tensor",
    "Linear", "Conv2D", "ResidualBlock", "flatten",
    "MLP", "MicroResNet",
    "mse_loss", "sgd",
    "train", "validate",
    "test",
    "mlp_dataset", "cnn_dataset", "mlp_test_dataset", "cnn_test_dataset",
]
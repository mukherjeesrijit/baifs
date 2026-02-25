# data.py
from .tensor import tensor

def mlp_dataset():
    X = [
        tensor([1.0, 2.0, 3.0]),
        tensor([2.0, 1.0, 0.5]),
        tensor([0.5, 1.5, 2.5]),
    ]
    y = [
        tensor([0.5, 1.5]),
        tensor([1.0, 0.0]),
        tensor([0.0, 1.0]),
    ]
    return X, y

def cnn_dataset(H=4, W=4):
    X = [
        [[[float(i*W + j + 1) for j in range(W)] for i in range(H)]]
    ]
    y = [tensor([1.0, 0.0])]
    return X, y

def mlp_test_dataset():
    X = [
        tensor([1.5, 2.5, 3.5]),
        tensor([0.1, 0.2, 0.3]),
    ]
    y = [
        tensor([1.0, 0.5]),
        tensor([0.1, 0.9]),
    ]
    return X, y

def cnn_test_dataset(H=4, W=4):
    X = [
        [[[float(i*W + j + 2) for j in range(W)] for i in range(H)]]
    ]
    y = [tensor([0.0, 1.0])]
    return X, y
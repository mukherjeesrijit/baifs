# layers.py
import math
import random

from .tensor import TensorGrad, ScalarGrad, scalar, tensor


class Module:
    def parameters(self):
        params = []

        def collect(obj):
            if isinstance(obj, TensorGrad):
                params.extend(obj.scalars())
            elif isinstance(obj, Module):
                for attr in obj.__dict__.values():
                    collect(attr)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, (list, tuple, set)):
                for v in obj:
                    collect(v)

        for attr in self.__dict__.values():
            collect(attr)
        return params


class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        self.W = tensor([[random.uniform(-scale, scale) for _ in range(in_features)]
                          for _ in range(out_features)])
        self.b = tensor([0.0 for _ in range(out_features)])

    def forward(self, x):
        return self.W.dot(x) + self.b


class Conv2D(Module):
    def __init__(self, in_ch, out_ch, k):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        scale = 1.0 / math.sqrt(in_ch * k * k) if in_ch * k * k > 0 else 0.0
        self.weight = tensor([
            [[random.uniform(-scale, scale) for _ in range(k)]
             for _ in range(k)]
            for _ in range(out_ch * in_ch)
        ])
        self.bias = tensor([0.0 for _ in range(out_ch)])

    def _kernel_scalar(self, oc, ic, ki, kj):
        idx = oc * self.in_ch + ic
        return self.weight.data[idx].data[ki].data[kj]._scalar()

    def forward(self, x):
        H, W = len(x[0]), len(x[0][0])
        H_, W_ = H - self.k + 1, W - self.k + 1
        out = []
        for oc in range(self.out_ch):
            b_oc = self.bias.data[oc]._scalar()
            grid = []
            for i in range(H_):
                row = []
                for j in range(W_):
                    s = scalar(0.0)
                    for ic in range(self.in_ch):
                        for ki in range(self.k):
                            for kj in range(self.k):
                                s = s + x[ic][i+ki][j+kj] * self._kernel_scalar(oc, ic, ki, kj)
                    s = s + b_oc
                    row.append(s)
                grid.append(row)
            out.append(grid)
        return out


class ResidualBlock(Module):
    def __init__(self, channels, k=3):
        assert k % 2 == 1
        self.conv1 = Conv2D(channels, channels, k)
        self.conv2 = Conv2D(channels, channels, k)
        self.pad = k // 2

    def forward(self, x):
        out = self.conv1.forward(_pad_input(x, self.pad))
        out = [[[v.relu() for v in row] for row in ch] for ch in out]
        out2 = self.conv2.forward(_pad_input(out, self.pad))
        skip = [
            [[x[ic][i][j] + out2[ic][i][j] for j in range(len(x[ic][0]))]
             for i in range(len(x[ic]))]
            for ic in range(len(x))
        ]
        return skip


def _pad_channel(ch, pad):
    H, W = len(ch), len(ch[0])
    top_bot = [[scalar(0.0) for _ in range(W + 2 * pad)] for _ in range(pad)]
    mid = [
        [scalar(0.0) for _ in range(pad)] + list(row) + [scalar(0.0) for _ in range(pad)]
        for row in ch
    ]
    return top_bot + mid + top_bot


def _pad_input(x, pad):
    return [_pad_channel(ch, pad) for ch in x]


def flatten(x):
    out = []
    for ch in x:
        for row in ch:
            for v in row:
                out.append(TensorGrad(v))
    return TensorGrad(out)
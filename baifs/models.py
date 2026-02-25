# models.py
from .layers import Linear, ResidualBlock, flatten, Module

class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.fc1.forward(x).relu()
        return self.fc2.forward(h)

class MicroResNet(Module):
    def __init__(self, in_ch, spatial_h=None, spatial_w=None, num_classes=2, H=None, W=None):
        if spatial_h is None:
            spatial_h = H
        if spatial_w is None:
            spatial_w = W
        if spatial_h is None or spatial_w is None:
            raise ValueError("MicroResNet requires spatial_h/spatial_w (or H/W).")

        self.res1 = ResidualBlock(in_ch, k=3)
        self.fc = Linear(in_ch * spatial_h * spatial_w, num_classes)

    def forward(self, x):
        out = self.res1.forward(x)
        out_flat = flatten(out)
        return self.fc.forward(out_flat)

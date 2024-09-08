import torch
from torch import nn


class ResidualLayers(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ...


class ReconNet(nn.Module):
    def __init__(self, inc=3, outc=3):
        super().__init__()
        self.rec = nn.Sequential(
            nn.Conv2d(inc, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, outc, 1),
        )

    def forward(self, x):
        return self.rec(x)

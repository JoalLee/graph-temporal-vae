"""Depthwise temporal convolution and window-token feed-forward blocks."""

import torch.nn as nn


class DepthwiseTCN(nn.Module):
    """Channel-independent dilated temporal feature extraction."""

    def __init__(self, channels, num_layers=3, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=channels,
                        bias=True,
                    ),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WindowTokenFFN(nn.Module):
    """Residual feed-forward network in window-token space."""

    def __init__(self, width, mult=4, dropout=0.1):
        super().__init__()
        width = int(width)
        hidden = int(mult) * width
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x + self.net(x))

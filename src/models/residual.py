import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, channels, hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock(channels, hidden) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

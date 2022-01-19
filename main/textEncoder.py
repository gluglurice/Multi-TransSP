import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Linearly transforms_train text.
    """
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x):
        x = self.weight * x + self.bias
        return x

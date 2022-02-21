"""
TextEncoder class in the model.

Author: Han
"""
import torch
import torch.nn as nn

import Chauhan.mainConfig as mc


class TextEncoder(nn.Module):
    """
    Encode text.
    """
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(mc.text_len, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.MLP(x)
        return x


if __name__ == '__main__':
    # print(TextEncoder())
    text_encoder = TextEncoder().to(mc.device)
    text = torch.rand(1, mc.text_len, dtype=torch.float32).to(mc.device)
    text_feature = text_encoder(text)
    print(text_feature.shape)

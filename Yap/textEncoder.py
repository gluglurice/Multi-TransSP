"""
TextEncoder class in the model.

Author: Han
"""
import torch
import torch.nn as nn

import Yap.mainConfig as mc


class TextEncoder(nn.Module):
    """
    Encode text.
    """
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.fc = nn.Linear(mc.text_len, mc.text_len)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # print(TextEncoder())
    text_encoder = TextEncoder().to(mc.device)
    text = torch.rand(1, mc.text_len, dtype=torch.float32).to(mc.device)
    text_feature = text_encoder(text)
    print(text_feature.shape)

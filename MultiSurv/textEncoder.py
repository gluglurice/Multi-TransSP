"""
TextEncoder class in the model.

Author: Han
"""
import torch
import torch.nn as nn

import MultiSurv.mainConfig as mc


class TextEncoder(nn.Module):
    """
    Encode text.
    """
    def __init__(self, text_len=mc.text_len, feature_channel=1000):
        super(TextEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(text_len, feature_channel),
            nn.LayerNorm(feature_channel),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(feature_channel, feature_channel),
            nn.LayerNorm(feature_channel),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(feature_channel, feature_channel),
            nn.LayerNorm(feature_channel),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(feature_channel, feature_channel),
            nn.LayerNorm(feature_channel),
            nn.ReLU(),

            nn.Dropout(),
            nn.Linear(feature_channel, feature_channel),
            nn.LayerNorm(feature_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # print(TextEncoder())
    text_encoder = TextEncoder(text_len=mc.text_len, feature_channel=1000).to(mc.device)
    text = torch.rand(1, mc.text_len, dtype=torch.float32).to(mc.device)
    text_feature = text_encoder(text)
    print(text_feature.shape)

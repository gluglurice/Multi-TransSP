"""
TransformerEncoder class in the model.

Author: Han
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce

import mainConfig as mc


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder from pytorch official.
    """
    def __init__(self, d_model=mc.d_model, nhead=mc.nhead, num_layers=mc.num_layers,
                 batch_size=mc.batch_size, num_patches=None, fusion_feature_channel=None):
        super(TransformerEncoder, self).__init__()

        self.batch_size = batch_size

        self.patch_to_embedding = nn.Linear(fusion_feature_channel, d_model)
        self.additional_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(batch_size, num_patches + 1, d_model))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer, num_layers=num_layers)

        self.fc_transformer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.patch_to_embedding(x)
        additional_tokens = repeat(self.additional_token, '() n d -> b n d', b=self.batch_size)
        x = torch.cat((additional_tokens, x), dim=1) + self.pos_embedding
        x = rearrange(x, 'b n d -> n b d')
        x = self.transformer_encoder(x)
        x = rearrange(x, 'n b d -> b n d')
        x = reduce(x, 'b n d -> b d', 'mean')
        x = self.fc_transformer(x)

        return x


if __name__ == '__main__':
    # print(TransformerEncoder())
    transformer_encoder = TransformerEncoder(
        d_model=mc.d_model, nhead=mc.nhead, num_layers=mc.num_layers,
        num_patches=121, fusion_feature_channel=524).to(mc.device)
    input_feature = torch.rand(8, 524, 11, 11, dtype=torch.float32).to(mc.device)
    output_feature = transformer_encoder(input_feature)
    print(output_feature.shape)

"""
Model class for predicting survivals.

Author: Han
"""
from math import ceil
import torch
from torch import nn
from einops import repeat

from main.resNetEncoder import ResNetEncoder
from main.textEncoder import TextEncoder
from main.transformerEncoder import TransformerEncoder
import main.mainConfig as mc


class Model(nn.Module):
    """
    The whole model.
    """

    def __init__(self, max_valid_slice_num, is_image=True, is_text=True, is_position=True, is_transformer=True):
        """
        :param max_valid_slice_num: max_valid_slice_num in dataset
        :param is_text: whether or not add text vector
        :param is_position: whether or not add spatial position vector
        :param is_transformer: whether or not add transformer_encoder
        """
        super(Model, self).__init__()
        self.max_valid_slice_num = max_valid_slice_num
        self.is_image = is_image
        self.is_text = is_text
        self.is_position = is_position
        self.is_transformer = is_transformer
        if self.is_image:
            self.image_encoder = ResNetEncoder()
        if self.is_text:
            self.text_encoder = TextEncoder()
        self.image_side_length, self.fusion_feature_channel = self.get_channel_num()
        self.num_patches = self.image_side_length ** 2
        if is_position:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 1, mc.size, mc.size))
        if is_transformer:
            self.transformer_encoder = TransformerEncoder(
                d_model=mc.d_model, nhead=mc.nhead, num_layers=mc.num_layers,
                num_patches=self.num_patches, fusion_feature_channel=self.fusion_feature_channel)
        else:
            self.fc = nn.Linear(self.fusion_feature_channel, 1)
        self.fc_survival = nn.Linear(max_valid_slice_num, mc.survivals_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image3D=None, text=None):
        """
        For each patient, go through the whole model, and output the predicted survival.
        :param image3D: image3D[0] in data batch of one patient, shape of [n 1 h w]
        :param text: text in data batch of one patient, shape of [1 length]
        """
        if self.is_image:
            survival_list = []
            text_feature = None
            """For each image batch of this patient."""
            for i in range(ceil(image3D.shape[0] / mc.batch_size)):
                image_batch = image3D[i * mc.batch_size:(i+1) * mc.batch_size]
                x_list = []

                if self.is_position:
                    pos_embedding = repeat(self.pos_embedding, '() c h w -> b c h w', b=image_batch.shape[0])
                    image_batch = image_batch + pos_embedding

                image_feature = self.image_encoder(image_batch)

                if not self.is_transformer:
                    image_feature = self.image_encoder.model.avgpool(image_feature)
                x_list.append(image_feature)

                if self.is_text and text_feature is None:
                        """3D expand text feature"""
                        text_feature = self.text_encoder(text).unsqueeze(-1).unsqueeze(-1)
                        text_feature = text_feature.expand(image_feature.shape[0], text.shape[1],
                                                           image_feature.shape[-2], image_feature.shape[-1])
                if text_feature is not None:
                    if image_feature.shape[0] != text_feature.shape[0]:
                        text_feature = text_feature[:image_feature.shape[0]]
                    x_list.append(text_feature)

                x = torch.cat(x_list, dim=1)
                if self.is_transformer:
                    x = self.transformer_encoder(x)
                else:
                    x = self.fc(x.squeeze(-1).squeeze(-1))
                x = self.sigmoid(x)
                survival_list.append(x)

            """Supplement zeros to adapt to max_valid_slice_num, and return the 4 survivals."""
            zeros = torch.zeros([self.max_valid_slice_num - image3D.shape[0], 1],
                                dtype=torch.float32).to(mc.device)
            survival_list.append(zeros)
            survivals = torch.cat(survival_list, 0).permute([1, 0])
            survivals = self.fc_survival(survivals)
            survivals = self.sigmoid(survivals)
            return survivals
        else:
            if self.is_text:
                x = self.text_encoder(text).unsqueeze(-1).unsqueeze(-1)
                if self.is_transformer:
                    x = self.transformer_encoder(x)
                else:
                    x = self.fc(x.squeeze(-1).squeeze(-1))
                survivals = self.fc_survival(x)
                survivals = self.sigmoid(survivals)
                return survivals
            else:
                return None

    def get_channel_num(self):
        """Get fusion_feature_channel and the num_patches of the convolved feature map."""
        with torch.no_grad():
            resnet_encoder = self.image_encoder.to(mc.device)
            image = torch.rand(1, 1, mc.size, mc.size, dtype=torch.float32).to(mc.device)
            image_feature = resnet_encoder(image)
            if not self.is_transformer:
                image_feature = self.image_encoder.model.avgpool(image_feature)

            image_side_length = image_feature.shape[2]

            fusion_feature_channel = image_feature.shape[1]
            if self.is_text:
                fusion_feature_channel += mc.text_len

            return image_side_length, fusion_feature_channel


if __name__ == '__main__':
    model = Model(85, is_text=mc.is_text, is_position=mc.is_position, is_transformer=mc.is_transformer)
    image3D = torch.rand(2, 1, mc.size, mc.size, dtype=torch.float32)
    text = torch.rand(1, 12, dtype=torch.float32)
    predicted_survival = model(image3D=image3D, text=text)
    print(predicted_survival.shape)
    print(predicted_survival)

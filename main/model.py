"""
Model class for predicting survivals.

Author: Han
"""
import torch
from torch import nn

from main.resNetEncoder import ResNetEncoder
from main.textEncoder import TextEncoder
from main.transformerEncoder import TransformerEncoder
import main.mainConfig as mc


class Model(nn.Module):
    """
    The whole model.
    """

    def __init__(self, max_valid_slice_num, is_text=True, is_position=True, is_transformer=True):
        """
        :param max_valid_slice_num: max_valid_slice_num in dataset
        :param is_text: whether or not add text vector
        :param is_position: whether or not add spatial position vector
        :param is_transformer: whether or not add transformer_encoder
        """
        super(Model, self).__init__()
        self.max_valid_slice_num = max_valid_slice_num
        self.is_text = is_text
        self.is_position = is_position
        self.is_transformer = is_transformer
        self.image_encoder = ResNetEncoder()
        self.text_encoder = TextEncoder()
        self.num_patches, self.fusion_feature_channel = self.get_channel_num()
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
        :param image3D: image3D in data batch of one patient
        :param text: text in data batch of one patient
        """
        survival_list = []
        if image3D is not None:
            text_feature = None
            """For each image piece of this patient."""
            for i, image in enumerate(image3D[0]):
                x_list = []

                image = image.unsqueeze(0).unsqueeze(0)
                image_feature = self.image_encoder(image)
                if not self.is_transformer:
                    image_feature = self.avgpool(image_feature)
                x_list.append(image_feature)

                if self.is_text and text_feature is None:
                        """3D expand text feature"""
                        text_feature = self.text_encoder(text).unsqueeze(-1).unsqueeze(-1)
                        text_feature = text_feature.expand(text.shape[0], text.shape[1],
                                                           image_feature.shape[-2], image_feature.shape[-1])
                if text_feature is not None:
                    x_list.append(text_feature)

                # if self.is_position:
                #     position = [0] * self.max_valid_slice_num
                #     position[i] = 1
                #     position = torch.tensor(position, dtype=torch.float).unsqueeze(0).to(mc.device)
                #     """3D expand position feature"""
                #     position_feature = position.unsqueeze(-1).unsqueeze(-1)
                #     position_feature = position_feature.expand(position_feature.shape[0], position_feature.shape[1],
                #                                                image_feature.shape[-2], image_feature.shape[-1])
                #     x_list.append(position_feature)

                x = torch.cat(x_list, dim=1)
                if self.is_transformer:
                    x = self.transformer_encoder(x)
                else:
                    x = self.fc(x.squeeze().unsqueeze(0))
                x = self.sigmoid(x)
                survival_list.append(x)

            """Supplement zeros to adapt to max_valid_slice_num, and return the 4 survivals."""
            zeros = torch.zeros([1, self.max_valid_slice_num - len(survival_list)],
                                dtype=torch.float32).to(mc.device)
            survival_list.append(zeros)
            survivals = torch.cat(survival_list, 1)
            survivals = self.fc_survival(survivals)
            survivals = self.sigmoid(survivals)
            return survivals

    def get_channel_num(self):
        """Get fusion_feature_channel and the num_patches of the convolved feature map."""
        with torch.no_grad():
            resnet_encoder = self.image_encoder.to(mc.device)
            image = torch.rand(1, 1, mc.size, mc.size, dtype=torch.float32).to(mc.device)
            image_feature = resnet_encoder(image)
            if not self.is_transformer:
                image_feature = self.image_encoder.model.avgpool(image_feature)

            num_patches = image_feature.shape[2] * image_feature.shape[3]

            fusion_feature_channel = image_feature.shape[1]
            if self.is_text:
                fusion_feature_channel += mc.text_len

            return num_patches, fusion_feature_channel


if __name__ == '__main__':
    model = Model(85)
    image3D = torch.rand(1, 2, mc.size, mc.size, dtype=torch.float32)
    text = torch.rand(1, 12, dtype=torch.float32)
    predicted_survival = model(image3D=image3D, text=text)
    print(predicted_survival)

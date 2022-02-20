"""
YapModel class of Yap's for predicting survivals.

Author: Han
"""
from math import ceil
import torch
from torch import nn

from Yap.resNetEncoder import ResNetEncoder
from main.textEncoder import TextEncoder
import Yap.mainConfig as mc


class YapModel(nn.Module):
    """
    The whole model.
    """

    def __init__(self, max_valid_slice_num, is_text=True):
        """
        :param max_valid_slice_num: max_valid_slice_num in dataset
        :param is_text: whether or not add text vector
        """
        super(YapModel, self).__init__()
        self.max_valid_slice_num = max_valid_slice_num
        self.is_text = is_text
        self.image_encoder = ResNetEncoder()
        self.text_encoder = TextEncoder()

        self.feature_channel = self.get_channel_num()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_channel, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1),
            nn.ReLU()
        )

        self.fc_survivals = nn.Linear(max_valid_slice_num, mc.survivals_len)
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
            """For each image batch of this patient."""
            for i in range(ceil(image3D.shape[0] / mc.batch_size)):
                image_batch = image3D[i * mc.batch_size:(i+1) * mc.batch_size]
                x_list = []

                image_feature = self.image_encoder(image_batch)
                x_list.append(image_feature)

                if self.is_text and text_feature is None:
                        text_feature = self.text_encoder(text)
                        text_feature = text_feature.expand(image_feature.shape[0], text.shape[1])
                if text_feature is not None:
                    if image_feature.shape[0] != text_feature.shape[0]:
                        text_feature = text_feature[:image_feature.shape[0]]
                    x_list.append(text_feature)

                x = torch.cat(x_list, dim=1)
                x = self.fc(x)
                survival_list.append(x)

        """Supplement zeros to adapt to max_valid_slice_num, and return the 4 survivals."""
        zeros = torch.zeros([self.max_valid_slice_num - image3D.shape[0], 1],
                            dtype=torch.float32).to(mc.device)
        survival_list.append(zeros)
        survivals = torch.cat(survival_list, 0).permute([1, 0])
        survivals = self.fc_survivals(survivals)
        survivals = self.sigmoid(survivals)
        return survivals

    def get_channel_num(self):
        """Get fusion_feature_channel and the num_patches of the convolved feature map."""
        with torch.no_grad():
            resnet_encoder = self.image_encoder.to(mc.device)
            image = torch.rand(1, 1, 332, 332, dtype=torch.float32).to(mc.device)
            image_feature = resnet_encoder(image)

            feature_channel = image_feature.shape[1]
            if self.is_text:
                feature_channel += mc.text_len

            return feature_channel


if __name__ == '__main__':
    model = YapModel(85)
    image3D = torch.rand(2, 1, 332, 332, dtype=torch.float32)
    text = torch.rand(1, 12, dtype=torch.float32)
    predicted_survivals = model(image3D=image3D, text=text)
    print(predicted_survivals)

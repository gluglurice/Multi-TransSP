"""
ChauhanModel class of Chauhan's for predicting survivals.

Author: Han
"""
from math import ceil
import torch
from torch import nn

from Chauhan.resNetEncoder import ResNetEncoder
from Chauhan.textEncoder import TextEncoder
import Chauhan.mainConfig as mc


class ChauhanModel(nn.Module):
    """
    The whole model.
    """

    def __init__(self, max_valid_slice_num):
        """
        :param max_valid_slice_num: max_valid_slice_num in dataset
        """
        super(ChauhanModel, self).__init__()
        self.max_valid_slice_num = max_valid_slice_num
        self.image_encoder = ResNetEncoder()
        self.text_encoder = TextEncoder()

        self.image_feature_channel, self.text_feature_channel = self.get_channel_num()
        self.fc_image = nn.Sequential(
            nn.Linear(self.image_feature_channel, 1),
            nn.ReLU()
        )
        self.fc_text = nn.Sequential(
            nn.Linear(self.text_feature_channel, 1),
            nn.ReLU()
        )
        self.fc_survivals = nn.Sequential(
            nn.Linear(max_valid_slice_num, mc.survivals_len),
            nn.Sigmoid()
        )

    def forward(self, image3D=None, text=None):
        """
        For each patient, go through the whole model, and output the predicted survival.
        :param image3D: image3D in data batch of one patient
        :param text: text in data batch of one patient
        """
        image_feature = None
        image_feature_list = []
        text_feature = None
        survivals_from_image_list = []
        predicted_survivals_from_image = None
        predicted_survivals_from_text = None

        if text is not None:
            text_feature = self.text_encoder(text)
            predicted_survivals_from_text = self.fc_text(text_feature)

        if image3D is not None:
            """For each image batch of this patient."""
            for i in range(ceil(image3D.shape[0] / mc.batch_size)):
                image_batch = image3D[i * mc.batch_size:(i + 1) * mc.batch_size]
                image_feature = self.image_encoder(image_batch)
                image_feature_list.append(image_feature)
            image_feature = torch.cat(image_feature_list, dim=0)
            predicted_survivals_from_image = self.fc_image(image_feature)
            survivals_from_image_list.append(predicted_survivals_from_image)

            """Supplement zeros to adapt to max_valid_slice_num, and return the 4 survivals."""
            zeros = torch.zeros([self.max_valid_slice_num - image3D.shape[0], 1],
                                dtype=torch.float32).to(mc.device)
            survivals_from_image_list.append(zeros)
            predicted_survivals_from_image = torch.cat(survivals_from_image_list, 0).permute([1, 0])
            predicted_survivals_from_image = self.fc_survivals(predicted_survivals_from_image)

        return image_feature, text_feature, predicted_survivals_from_image, predicted_survivals_from_text

    def get_channel_num(self):
        """Get feature channels of the convolved feature map."""
        with torch.no_grad():
            image_encoder = self.image_encoder.to(mc.device)
            text_encoder = self.text_encoder.to(mc.device)
            image = torch.rand(1, 1, mc.size, mc.size, dtype=torch.float32).to(mc.device)
            text = torch.rand(1, mc.text_len, dtype=torch.float32)
            image_feature = image_encoder(image)
            text_feature = text_encoder(text)

            image_feature_channel = image_feature.shape[1]
            text_feature_channel = text_feature.shape[1]

            return image_feature_channel, text_feature_channel


if __name__ == '__main__':
    model = ChauhanModel(85)
    image3D = torch.rand(1, 2, 1, mc.size, mc.size, dtype=torch.float32)
    text = torch.rand(1, 12, dtype=torch.float32)
    image_feature, text_feature, predicted_survivals_from_image, predicted_survivals_from_text \
        = model(image3D=image3D[0], text=text)
    print(image_feature.shape)
    print(text_feature.shape)
    print(predicted_survivals_from_image.shape)
    print(predicted_survivals_from_text.shape)

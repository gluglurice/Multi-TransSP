"""
YapModel class of Yap's for predicting survivals.

Author: Han
"""
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
        self.MLP = nn.Sequential(
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
            """For each image piece of this patient."""
            for i, image in enumerate(image3D[0]):
                x_list = []

                image = image.unsqueeze(0).unsqueeze(0)
                image_feature = self.image_encoder(image)
                x_list.append(image_feature)

                if text is not None:
                    x_list.append(text)

                x = torch.cat(x_list, dim=1)
                x = self.MLP(x)
                survival_list.append(x)

            """Supplement zeros to adapt to max_valid_slice_num, and return the 4 survivals."""
            zeros = torch.zeros([1, self.max_valid_slice_num - len(survival_list)], dtype=torch.float32).to(mc.device)
            survival_list.append(zeros)
            survivals = torch.cat(survival_list, 1)
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
    image3D = torch.rand(1, 2, 332, 332, dtype=torch.float32)
    text = torch.rand(1, 12, dtype=torch.float32)
    mask = torch.ones([1, 256]).bool()
    predicted_survivals = model(image3D=image3D, text=text, mask=mask)
    print(predicted_survivals)

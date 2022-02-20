"""
ResNetEncoder class in the model.

Author: Han
"""
import glob

import torch
import torch.nn as nn
from torchvision import models

import Chauhan.mainConfig as mc


class ResNetEncoder(nn.Module):
    """
    ResNet from pytorch official.
    """
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        """Load pretrained resnet."""
        self.model = models.resnet50(pretrained=False)
        if len(glob.glob(mc.model_path_reg)) == 0:
            self.model.load_state_dict(torch.load(mc.model_resnet_path, map_location=mc.device))

        """Modify conv1."""
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x).squeeze(-1).squeeze(-1)

        return x


if __name__ == '__main__':
    # print(ResNetEncoder())
    resnet_encoder = ResNetEncoder().to(mc.device)
    image = torch.rand(1, 1, mc.size, mc.size, dtype=torch.float32).to(mc.device)
    image_feature = resnet_encoder(image)
    print(image_feature.shape)

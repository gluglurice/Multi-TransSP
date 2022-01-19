import glob

import torch
import torch.nn as nn
from torchvision import models

import mainConfig as mc


class ResNetEncoder(nn.Module):
    """
    ResNet50 from pytorch official.
    """
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        """Load pretrained resnet50."""
        self.model = models.resnet50(pretrained=False)
        if len(glob.glob(mc.model_path_reg)) == 0:
            self.model.load_state_dict(torch.load(mc.model_resnet50_path, map_location=mc.device))

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

        return x

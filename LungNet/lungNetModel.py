"""
LungNetModel class of Yap's for predicting survivals.

Author: Han
"""
import torch
from torch import nn

import LungNet.mainConfig as mc


class LungNetModel(nn.Module):
    """
    The whole model.
    """

    def __init__(self, max_valid_slice_num):
        """
        :param max_valid_slice_num: max_valid_slice_num in dataset
        """
        super(LungNetModel, self).__init__()
        self.max_valid_slice_num = max_valid_slice_num

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),

            nn.Conv3d(64, 256, kernel_size=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image3D=None):
        """
        For each patient, go through the whole model, and output the predicted survival.
        :param image3D: image3D in data batch of one patient, shape of [n c d h w].
        """
        if image3D is not None:
            """Supplement zeros to adapt to max_valid_slice_num, and return the survivals."""
            zeros = torch.zeros(
                [image3D.shape[0], 1, self.max_valid_slice_num - image3D.shape[2],
                 image3D.shape[3], image3D.shape[4]],
                dtype=torch.float32).to(mc.device)
            """For each image batch of this patient."""
            image3D = torch.cat([image3D, zeros], dim=2)

            x = self.conv3d(image3D)
            x = self.fc(x.squeeze(-1).squeeze(-1).squeeze(-1))
            return x
        else:
            return None


if __name__ == '__main__':
    model = LungNetModel(85)
    image3D = torch.rand(mc.patient_batch_size, 1, 2, mc.size, mc.size, dtype=torch.float32)
    predicted_survivals = model(image3D=image3D)
    print(predicted_survivals.shape)
    print(predicted_survivals)

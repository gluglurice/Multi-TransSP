"""
DeepSurvModel class of Yap's for predicting survivals.

Author: Han
"""
import torch
from torch import nn

import DeepSurv.mainConfig as mc


class DeepSurvModel(nn.Module):
    """
    The whole model.
    """

    def __init__(self, text_len=mc.text_len):
        super(DeepSurvModel, self).__init__()
        self.text_len = text_len

        self.fc = nn.Sequential(
            nn.Linear(self.text_len, 1),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, text=None):
        """
        For each patient, go through the whole model, and output the predicted survival.
        :param text: text in data batch of one patient
        """
        """For each image batch of this patient."""
        survivals = self.fc(text)

        return survivals


if __name__ == '__main__':
    model = DeepSurvModel()
    text = torch.rand(1, 12, dtype=torch.float32)
    predicted_survivals = model(text=text)
    print(predicted_survivals.shape)
    print(predicted_survivals)

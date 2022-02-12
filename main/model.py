"""
Model class for predicting survivals.

Author: Han
"""
import torch
from torch import nn

from resNetEncoder import ResNetEncoder
from textEncoder import TextEncoder
from fastformer import Fastformer
import mainConfig as mc


class Model(nn.Module):
    """
    The whole model.
    """

    def __init__(self, max_valid_slice_num, is_position=True):
        """
        :param max_valid_slice_num: max_valid_slice_num in dataset
        :param is_position: whether or not add spatial position vector
        """
        super(Model, self).__init__()
        self.max_valid_slice_num = max_valid_slice_num
        self.is_position = is_position
        self.image_encoder = ResNetEncoder()
        self.text_encoder = TextEncoder()
        self.dimension, self.conv_1_1_channel = self.get_channel_num()
        self.fastformer = Fastformer(num_tokens=mc.sequence_length, dim=self.dimension,
                                     depth=2, max_seq_len=256, absolute_pos_emb=True)
        self.conv_1_1 = nn.Conv2d(self.conv_1_1_channel, mc.sequence_length,
                                  kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.fc_fastformer = nn.Linear(mc.sequence_length, 1)
        self.fc_survivals = nn.Linear(max_valid_slice_num, mc.survivals_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image3D=None, text=None, mask=None):
        """
        For each patient, go through the whole model, and output the predicted survival.
        :param image3D: image3D in data batch of one patient
        :param text: text in data batch of one patient
        :param mask: mask of fastformer
        """
        survival_list = []
        if image3D is not None:
            text_feature = None
            """For each image piece of this patient."""
            for i, image in enumerate(image3D[0]):
                x_list = []

                image = image.unsqueeze(0).unsqueeze(0)
                image_feature = self.image_encoder(image)
                x_list.append(image_feature)

                if text is not None and text_feature is None:
                    """3D expand text feature"""
                    text_feature = self.text_encoder(text).unsqueeze(-1).unsqueeze(-1)
                    text_feature = text_feature.expand(text.shape[0], text.shape[1],
                                                       image_feature.shape[-2], image_feature.shape[-1])
                x_list.append(text_feature)

                if self.is_position is True:
                    position = [0] * self.max_valid_slice_num
                    position[i] = 1
                    position = torch.tensor(position, dtype=torch.float).unsqueeze(0)
                    """3D expand position feature"""
                    position_feature = position.unsqueeze(-1).unsqueeze(-1)
                    position_feature = position_feature.expand(position_feature.shape[0], position_feature.shape[1],
                                                               image_feature.shape[-2], image_feature.shape[-1])
                    x_list.append(position_feature)

                x = torch.cat(x_list, dim=1)
                x = self.conv_1_1(x)
                x = self.flatten(x)
                x = self.fastformer(x, mask=mask).squeeze(-1)
                x = self.fc_fastformer(x)
                x = self.sigmoid(x)
                survival_list.append(x)

            """Supplement zeros to adapt to max_valid_slice_num, and return the 4 survivals."""
            zeros = torch.zeros([1, self.max_valid_slice_num - len(survival_list)], dtype=torch.float32).to(mc.device)
            survival_list.append(zeros)
            survivals = torch.cat(survival_list, 1)
            survivals = self.fc_survivals(survivals)
            survivals = self.sigmoid(survivals)
            return survivals

        elif text is not None and image3D is None:
            pass

    def get_channel_num(self):
        """Get conv_1_1_channel and the dimension of the convolved feature map."""
        resnet_encoder = self.image_encoder.to(mc.device)
        image = torch.rand(1, 1, 332, 332, dtype=torch.float32).to(mc.device)
        image_feature = resnet_encoder(image)

        dimension = image_feature.shape[2] * image_feature.shape[3]

        conv_1_1_channel = image_feature.shape[1] + mc.text_len
        if self.is_position is True:
            conv_1_1_channel += self.max_valid_slice_num

        return dimension, conv_1_1_channel


if __name__ == '__main__':
    model = Model(85)
    image3D = torch.rand(1, 2, 332, 332, dtype=torch.float32)
    text = torch.rand(1, 12, dtype=torch.float32)
    mask = torch.ones([1, 256]).bool()
    predicted_survivals = model(image3D=image3D, text=text, mask=mask)
    print(predicted_survivals)

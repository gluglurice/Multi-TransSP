"""
This file aims to
test the Yap model for predicting survival.

Author: Han
"""
import os
import glob
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from lifelines.utils import concordance_index

from myDataset import MyDataset
from main.model import Model
import main.mainConfig as mc


def test():
    summary_path = f'./summary_test_{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    summary_writer_test = SummaryWriter(summary_path + '/test')

    """(1) Prepare data."""
    test_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='test',
                         ki=0, k=mc.k, transform=mc.transforms_test, rand=True)
    test_loader = DataLoader(test_set, batch_size=mc.batch_size,
                             shuffle=True, num_workers=mc.num_workers)
    max_valid_slice_num = test_set.max_valid_slice_num

    """(2) Prepare Network."""
    """Model."""
    model = Model(max_valid_slice_num, is_text=mc.is_text, is_position=mc.is_position,
                  is_fastformer=mc.is_fastformer).to(mc.device)

    if len(glob.glob(mc.model_path_reg)) > 0:
        model_path = sorted(glob.glob(mc.model_path_reg),
                            key=lambda name: int(name.split('_')[-1].split('.')[0]))[-1]
        model.load_state_dict(torch.load(model_path, map_location=mc.device))

    """Loss."""
    criterion_MSE = nn.MSELoss()

    """(3) Start testing."""
    with torch.no_grad():
        model.eval()
        test_tqdm = tqdm(test_loader)

        label_survivals_history = []
        predicted_survivals_history = []
        loss_history = []
        cos_similarity_history = []

        for i, batch in enumerate(test_tqdm):
            """Data."""
            image3D = batch['image3D'].to(mc.device)
            text = batch['text'].to(mc.device)
            label_survivals = batch['survivals'].to(mc.device)
            mask = torch.ones([1, mc.sequence_length]).bool().to(mc.device)

            """Predict."""
            predicted_survivals = model(image3D=image3D, text=text, mask=mask).to(mc.device)

            """Loss."""
            loss_survivals = criterion_MSE(predicted_survivals, label_survivals)
            cos_similarity = torch.cosine_similarity(predicted_survivals, label_survivals, dim=-1)

            test_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')

            label_survivals_array = np.array(label_survivals.squeeze())
            predicted_survivals_array = np.array(predicted_survivals.squeeze())
            loss_survivals_array = np.array(loss_survivals)
            cos_similarity_array = np.array(cos_similarity)

            summary_writer_test.add_scalar('MSE Loss', loss_survivals_array, i)
            summary_writer_test.add_scalar('Cos Similarity', cos_similarity_array, i)

            label_survivals_history.append(label_survivals_array)
            predicted_survivals_history.append(predicted_survivals_array)
            loss_history.append(loss_survivals_array)
            cos_similarity_history.append(cos_similarity_array)

        summary_writer_test.add_scalar('Mean MSE Loss', np.array(loss_history).mean(axis=0))
        summary_writer_test.add_scalar('Mean Cos Similarity', np.array(cos_similarity_history).mean(axis=0))

        c_index = np.array([concordance_index(
            np.array(label_survivals_history)[:, i],
            np.array(predicted_survivals_history)[:, i]) for i in range(4)]).mean(axis=0)

        summary_writer_test.add_scalar('Mean C Index', c_index)


if __name__ == '__main__':
    test()

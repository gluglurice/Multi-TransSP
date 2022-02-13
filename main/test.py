"""
This file aims to
test the model for predicting survival.

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

from myDataset import MyDataset
from model import Model
import mainConfig as mc


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
    model = Model(max_valid_slice_num, is_position=mc.is_position).to(mc.device)

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
        loss_history = []
        for i, batch in enumerate(test_tqdm):
            """Data."""
            image3D = batch['image3D'].to(mc.device)
            text = batch['text'].to(mc.device)
            label_survivals = batch['survivals'].to(mc.device)
            mask = torch.ones([1, mc.sequence_length]).bool().to(mc.device)

            """Predict."""
            predicted_survivals = model(image3D=image3D, text=text, mask=mask).to(mc.device)

            """Loss & Optimize."""
            loss_survivals = criterion_MSE(predicted_survivals, label_survivals).to(mc.device)

            """tqdm postfix."""
            test_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')
            loss_survivals = np.array(loss_survivals.detach().cpu())
            summary_writer_test.add_scalar('MSE Loss', loss_survivals, i)
            loss_history.append(loss_survivals)

        summary_writer_test.add_scalar('Mean MSE Loss', np.array(loss_history).mean(axis=0))


if __name__ == '__main__':
    test()

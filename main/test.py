"""
This file aims to
test the model for predicting survival.

Author: Han
"""
import os
import glob

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

    if not os.path.exists(mc.summary_path):
        os.makedirs(mc.summary_path)

    model_paths = []
    if len(glob.glob(mc.test_model_path_reg)) > 0:
        model_paths = sorted(glob.glob(mc.test_model_path_reg),
                             key=lambda name: int(name.split('.')[0].split('_')[-1]))

    summary_writer_test = SummaryWriter(mc.summary_path + '/test')

    """(1) Prepare data."""
    test_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='test',
                         ki=0, k=mc.k, transform=mc.transforms_test, rand=True)
    test_loader = DataLoader(test_set, batch_size=mc.batch_size,
                             shuffle=True, num_workers=mc.num_workers)

    max_valid_slice_num = test_set.max_valid_slice_num

    """(2) Prepare Network."""
    """Model."""
    model = Model(max_valid_slice_num, is_text=mc.is_text, is_position=mc.is_position,
                  is_transformer=mc.is_transformer).to(mc.device)

    if len(model_paths) > 0:
        model.load_state_dict(torch.load(model_paths[-1], map_location=mc.device))

    """Loss."""
    criterion_MSE = nn.MSELoss()

    """(3) Start testing."""
    with torch.no_grad():
        model.eval()
        test_tqdm = tqdm(test_loader, desc=f'Test', colour='#27ce82')

        label_survivals_history = []
        predicted_survivals_history = []
        loss_history = []
        cos_similarity_history = []

        for i, batch in enumerate(test_tqdm):
            """Data."""
            image3D = batch['image3D'].to(mc.device)
            text = batch['text'].to(mc.device)
            label_survivals = batch['survivals'].to(mc.device)

            """Predict."""
            predicted_survivals = model(image3D=image3D, text=text).to(mc.device)

            """Loss."""
            loss_survivals = criterion_MSE(predicted_survivals, label_survivals)
            cos_similarity = torch.cosine_similarity(predicted_survivals, label_survivals, dim=-1)

            test_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')

            label_survivals_array = np.array(label_survivals.squeeze(0).detach().cpu())
            predicted_survivals_array = np.array(predicted_survivals.squeeze(0).detach().cpu())
            loss_survivals_array = np.array(loss_survivals.detach().cpu())
            cos_similarity_array = np.array(cos_similarity.detach().cpu())

            label_survivals_history.append(label_survivals_array)
            predicted_survivals_history.append(predicted_survivals_array)
            loss_history.append(loss_survivals_array)
            cos_similarity_history.append(cos_similarity_array)

        c_index = np.array([concordance_index(
            np.array(label_survivals_history)[:, i],
            np.array(predicted_survivals_history)[:, i]) for i in range(4)]).mean(axis=0)

        loss_history_array_mean = np.array(loss_history).mean(axis=0)
        cos_similarity_history_array_mean = np.array(cos_similarity_history).mean(axis=0)

        summary_writer_test.add_scalar('MSE Loss', loss_history_array_mean)
        summary_writer_test.add_scalar('Cos Similarity', cos_similarity_history_array_mean)
        summary_writer_test.add_scalar('C Index', c_index)


if __name__ == '__main__':
    test()

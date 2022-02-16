"""
This file aims to
train the model for predicting survival.

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


def train():
    """KFold training."""

    if not os.path.exists(mc.summary_path):
        os.makedirs(mc.summary_path)

    summary_writer_train = SummaryWriter(mc.summary_path + f'/train')
    summary_writer_test = SummaryWriter(mc.summary_path + f'/test')

    """(1) Prepare data."""
    train_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='train_all',
                          ki=0, k=mc.k, transform=mc.transforms_train, rand=True)
    test_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='test',
                         ki=0, k=mc.k, transform=mc.transforms_train, rand=True)

    train_loader = DataLoader(train_set, batch_size=mc.batch_size,
                              shuffle=True, num_workers=mc.num_workers)
    test_loader = DataLoader(test_set, batch_size=mc.batch_size,
                             shuffle=True, num_workers=mc.num_workers)

    max_valid_slice_num = train_set.max_valid_slice_num

    """(2) Prepare Network."""
    """Model."""
    model = Model(max_valid_slice_num, is_text=mc.is_text, is_position=mc.is_position,
                  is_transformer=mc.is_transformer).to(mc.device)

    """Loss & Optimize."""
    criterion_MSE = nn.MSELoss()

    opt_model = torch.optim.SGD(model.parameters(), lr=mc.lr, momentum=0.9, weight_decay=mc.weight_decay)

    """(3) Start training."""
    for epoch in range(mc.epoch_start, mc.epoch_end):

        """Train."""
        model.train()
        train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}, Train', colour='#f14461')
        loss_train_history = []
        for i, batch in enumerate(train_tqdm):
            """Data."""
            image3D = batch['image3D'].to(mc.device)
            text = batch['text'].to(mc.device)
            label_survivals = batch['survivals'].to(mc.device)

            """Predict."""
            predicted_survivals = model(image3D=image3D, text=text).to(mc.device)

            """Loss & Optimize."""
            loss_survivals = criterion_MSE(predicted_survivals, label_survivals).to(mc.device)

            opt_model.zero_grad()
            loss_survivals.backward()
            opt_model.step()

            train_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')

            loss_train_history.append(np.array(loss_survivals.detach().cpu()))

        loss_train_history = np.array(loss_train_history)
        loss_train_history_mean = loss_train_history.mean(axis=0)

        summary_writer_train.add_scalar('MSE Loss', loss_train_history_mean, epoch + 1)

        """Test."""
        with torch.no_grad():
            model.eval()
            test_tqdm = tqdm(test_loader, desc=f'Epoch {epoch}, Test', colour='#27ce82')

            label_survivals_history = []
            predicted_survivals_history = []
            loss_test_history = []
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

                label_survivals_array = np.array(label_survivals.squeeze().detach().cpu())
                predicted_survivals_array = np.array(predicted_survivals.squeeze().detach().cpu())
                loss_survivals_array = np.array(loss_survivals.detach().cpu())
                cos_similarity_array = np.array(cos_similarity.detach().cpu())

                label_survivals_history.append(label_survivals_array)
                predicted_survivals_history.append(predicted_survivals_array)
                loss_test_history.append(loss_survivals_array)
                cos_similarity_history.append(cos_similarity_array)

            c_index = np.array([concordance_index(
                np.array(label_survivals_history)[:, i],
                np.array(predicted_survivals_history)[:, i]) for i in range(4)]).mean(axis=0)

            loss_test_history_mean = np.array(loss_test_history).mean(axis=0)
            cos_similarity_history_mean = np.array(cos_similarity_history).mean(axis=0)

            summary_writer_test.add_scalar('MSE Loss', loss_test_history_mean, epoch + 1)
            summary_writer_test.add_scalar('Cos Similarity', cos_similarity_history_mean, epoch + 1)
            summary_writer_test.add_scalar('C Index', c_index, epoch + 1)

            """Save model."""
            if loss_test_history_mean < mc.min_loss:
                mc.min_loss = loss_test_history_mean
                """Remove former models."""
                if len(glob.glob(mc.model_path_reg)) > 0:
                    model_path = sorted(glob.glob(mc.test_model_path_reg),
                                        key=lambda name: int(name.split('.')[0].split('_')[-1]))[-1]
                    if os.path.exists(model_path):
                        os.remove(model_path)
                """Save the model that has had the min loss so far."""
                if not os.path.exists(mc.model_path):
                    os.makedirs(mc.model_path)
                torch.save(model.state_dict(), f'{mc.model_path}/epoch_{epoch + 1}.pth')
            if epoch == mc.epoch_end - 1:
                """Reset min_loss for the next fold."""
                mc.min_loss = 1e10
                if not os.path.exists(mc.model_path):
                    os.makedirs(mc.model_path)
                torch.save(model.state_dict(), f'{mc.model_path}/epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    train()

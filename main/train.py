"""
This file aims to
train the Model for predicting survival.

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
from utils import LambdaLR


def train():
    """KFold training."""
    for ki in range(mc.k):

        print(f'Fold {ki + 1}/{mc.k}:')

        """(1) Prepare data."""
        train_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='train',
                              ki=ki, k=mc.k, transform=mc.transforms_train, rand=True)
        validate_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='validate',
                                 ki=ki, k=mc.k, transform=mc.transforms_train, rand=True)

        train_loader = DataLoader(train_set, batch_size=mc.batch_size,
                                  shuffle=True, num_workers=mc.num_workers)
        validate_loader = DataLoader(validate_set, batch_size=mc.batch_size,
                                     shuffle=True, num_workers=mc.num_workers)

        max_valid_slice_num = train_set.max_valid_slice_num

        """(2) Prepare Network."""
        """Model."""
        model = Model(max_valid_slice_num).to(mc.device)

        if len(glob.glob(mc.model_path_reg)) > 0:
            model_path = sorted(glob.glob(mc.model_path_reg),
                                key=lambda name: int(name.split('_')[-1].split('.')[0]))[-1]
            model.load_state_dict(torch.load(model_path, map_location=mc.device))
            """Continue the epoch number."""
            epoch_base = int(model_path.split('_')[-1].split('.')[0])
            mc.epoch_start += epoch_base
            mc.epoch_total += epoch_base

        """Loss & Optimize."""
        criterion_MSE = nn.MSELoss()

        opt_model = torch.optim.Adam(model.parameters(), lr=mc.lr, betas=(0.9, 0.9999))

        lr_scheduler_model = torch.optim.lr_scheduler.LambdaLR(
            opt_model, lr_lambda=LambdaLR(mc.epoch_total, mc.epoch_start, mc.epoch_decay).step)

        """(3) Start training."""

        # summary_path = f'./summary_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
        # if not os.path.exists(summary_path):
        #     os.makedirs(summary_path)
        # summary_writer_train = SummaryWriter(summary_path + '/train')
        # summary_writer_eval = SummaryWriter(summary_path + '/eval')

        for epoch in range(mc.epoch_start, mc.epoch_total):

            """Train."""
            model.train()
            train_tqdm = tqdm(train_loader)
            loss_history = []
            for i, batch in enumerate(train_tqdm):
                """Data."""
                image3D = batch['image3D'].to(mc.device)
                text = batch['text'].to(mc.device)
                label_survivals = batch['survivals'].to(mc.device)
                mask = torch.ones([1, mc.sequence_length]).bool().to(mc.device)

                """Predict."""
                predicted_survivals = model(image3D=image3D, text=text, mask=mask)

                """Loss & Optimize."""
                loss_survivals = criterion_MSE(predicted_survivals, label_survivals)

                opt_model.zero_grad()
                loss_survivals.backward()
                opt_model.step()

                """tqdm postfix."""
                train_tqdm.set_description(f'epoch {epoch}')
                train_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')
                loss_history.append(np.array(loss_survivals))

            loss_history = np.array(loss_history)
            summary_writer_train.add_scalar('Epoch Loss OS', loss_history.mean(axis=0).squeeze()[0])
            summary_writer_train.add_scalar('Epoch Loss MFS', loss_history.mean(axis=0).squeeze()[1])
            summary_writer_train.add_scalar('Epoch Loss RFS', loss_history.mean(axis=0).squeeze()[2])
            summary_writer_train.add_scalar('Epoch Loss PFS', loss_history.mean(axis=0).squeeze()[3])

            lr_scheduler_model.step()

            """Save model."""
            if ((epoch - mc.epoch_start - 19) % 20 == 0) or (epoch == mc.epoch_total - 1):
                torch.save(model.state_dict(), f'./model/model_epoch_{epoch + 1}.pth')

            # """Validate."""
            # with torch.no_grad():
            #     model.eval()
            #     validate_tqdm = tqdm(validate_loader)
            #     loss_history = []
            #     for i, batch in enumerate(validate_tqdm):
            #         """Data."""
            #         image3D = batch['image3D'].to(mc.device)
            #         text = batch['text'].to(mc.device)
            #         label_survivals = batch['survivals'].to(mc.device)
            #         mask = torch.ones([1, mc.sequence_length]).bool().to(mc.device)
            #
            #         """Predict."""
            #         predicted_survivals = model(image3D=image3D, text=text, mask=mask)
            #
            #         """Loss & Optimize."""
            #         loss_survivals = criterion_MSE(predicted_survivals, label_survivals)
            #
            #         train_tqdm.set_description(f'epoch {epoch}')
            #         train_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')
            #         loss_history.append(np.array(loss_survivals))
            #
            #     loss_history = np.array(loss_history)
            #     summary_writer_eval.add_scalar('Epoch Loss OS', loss_history.mean(axis=0).squeeze()[0])
            #     summary_writer_eval.add_scalar('Epoch Loss MFS', loss_history.mean(axis=0).squeeze()[1])
            #     summary_writer_eval.add_scalar('Epoch Loss RFS', loss_history.mean(axis=0).squeeze()[2])
            #     summary_writer_eval.add_scalar('Epoch Loss PFS', loss_history.mean(axis=0).squeeze()[3])


if __name__ == '__main__':
    train()

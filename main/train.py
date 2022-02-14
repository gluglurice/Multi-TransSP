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

from myDataset import MyDataset
from main.model import Model
import main.mainConfig as mc
from utils import LambdaLR


def train():
    """KFold training."""

    if not os.path.exists(mc.summary_path):
        os.makedirs(mc.summary_path)

    for ki in range(mc.k_start, mc.k):

        summary_writer_train = SummaryWriter(mc.summary_path + f'/train{ki}')
        summary_writer_eval = SummaryWriter(mc.summary_path + f'/eval{ki}')

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
        model = Model(max_valid_slice_num, is_text=mc.is_text, is_position=mc.is_position,
                      is_fastformer=mc.is_fastformer).to(mc.device)

        """Loss & Optimize."""
        criterion_MSE = nn.MSELoss()

        opt_model = torch.optim.Adam(model.parameters(), lr=mc.lr, betas=(0.9, 0.9999))

        lr_scheduler_model = torch.optim.lr_scheduler.LambdaLR(
            opt_model, lr_lambda=LambdaLR(mc.epoch_end, mc.epoch_start, mc.epoch_decay).step)

        """(3) Start training."""
        for epoch in range(mc.epoch_start, mc.epoch_end):

            """Train."""
            model.train()
            train_tqdm = tqdm(train_loader, desc=f'Fold {ki + 1}, epoch {epoch}, Train', colour='#f14461')
            loss_train_history = []
            for i, batch in enumerate(train_tqdm):
                """Data."""
                image3D = batch['image3D'].to(mc.device)
                text = batch['text'].to(mc.device)
                label_survivals = batch['survivals'].to(mc.device)
                mask = torch.ones([1, mc.sequence_length]).bool().to(mc.device)

                """Predict."""
                predicted_survivals = model(image3D=image3D, text=text, mask=mask).to(mc.device)

                """Loss & Optimize."""
                loss_survivals = criterion_MSE(predicted_survivals, label_survivals).to(mc.device)

                opt_model.zero_grad()
                loss_survivals.backward()
                opt_model.step()

                train_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')

                loss_train_history.append(np.array(loss_survivals.detach().cpu()))

            loss_train_history = np.array(loss_train_history)
            loss_train_history_mean = loss_train_history.mean(axis=0)

            summary_writer_train.add_scalar('Epoch MSE Loss', loss_train_history_mean, epoch)

            lr_scheduler_model.step()

            """Eval."""
            with torch.no_grad():
                model.eval()
                validate_tqdm = tqdm(validate_loader, desc=f'Fold {ki + 1}, epoch {epoch}, Eval', colour='#4286e7')
                loss_eval_history = []
                for i, batch in enumerate(validate_tqdm):
                    """Data."""
                    image3D = batch['image3D'].to(mc.device)
                    text = batch['text'].to(mc.device)
                    label_survivals = batch['survivals'].to(mc.device)
                    mask = torch.ones([1, mc.sequence_length]).bool().to(mc.device)

                    """Predict."""
                    predicted_survivals = model(image3D=image3D, text=text, mask=mask).to(mc.device)

                    """Loss & Optimize."""
                    loss_survivals = criterion_MSE(predicted_survivals, label_survivals).to(mc.device)

                    validate_tqdm.set_postfix(loss_survivals=f'{loss_survivals.item():.4f}')

                    loss_eval_history.append(np.array(loss_survivals.detach().cpu()))

                loss_eval_history = np.array(loss_eval_history)
                loss_eval_history_mean = loss_eval_history.mean(axis=0)

                summary_writer_eval.add_scalar('Epoch MSE Loss', loss_eval_history_mean, epoch)

                """Save model."""
                if loss_eval_history_mean < mc.min_loss:
                    mc.min_loss = loss_eval_history_mean
                    """Remove former models."""
                    if len(glob.glob(mc.model_path_reg)) > 0:
                        model_path = sorted(glob.glob(mc.model_path_reg),
                                            key=lambda name: int(name.split('_')[-3]))[-1]
                        if int(model_path.split('_')[-3]) == ki+1:
                            os.remove(model_path)
                    """Save the model that has had the min loss so far."""
                    torch.save(model.state_dict(), f'{mc.model_path}/fold_{ki+1}_epoch_{epoch}.pth')
                if epoch == mc.epoch_end - 1:
                    """Reset min_loss for the next fold."""
                    mc.min_loss = 1e10


if __name__ == '__main__':
    train()

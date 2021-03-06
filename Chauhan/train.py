"""
This file aims to
train the Chauhan's model for predicting survival.

Author: Han
"""
import os
import glob

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
from tensorboardX import SummaryWriter
from lifelines.utils import concordance_index

from myDataset import MyDataset
from Chauhan.chauhanModel import ChauhanModel
import Chauhan.mainConfig as mc


def train():
    if not os.path.exists(mc.summary_path):
        os.makedirs(mc.summary_path)

    summary_writer_train = SummaryWriter(mc.summary_path + f'/train')
    summary_writer_test = SummaryWriter(mc.summary_path + f'/test')

    """(1) Prepare data."""
    train_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='train_all',
                          ki=0, k=mc.k, transform=mc.transforms_train, rand=True)
    test_set = MyDataset(root=mc.data_path, excel_path=mc.excel_path, mode='test',
                         ki=0, k=mc.k, transform=mc.transforms_train, rand=True)

    train_loader = DataLoader(train_set, batch_size=mc.patient_batch_size,
                              shuffle=True, num_workers=mc.num_workers)
    test_loader = DataLoader(test_set, batch_size=mc.patient_batch_size,
                             shuffle=True, num_workers=mc.num_workers)

    max_valid_slice_num = train_set.max_valid_slice_num

    """(2) Prepare Network."""
    """Model."""
    model = ChauhanModel(max_valid_slice_num).to(mc.device)

    """Loss & Optimize."""
    criterion_MSE = nn.MSELoss()

    opt_model = torch.optim.SGD(model.parameters(), lr=mc.lr, momentum=0.9, weight_decay=mc.weight_decay)
    lr_scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=20, eta_min=1e-5)

    """(3) Start training."""
    for epoch in range(mc.epoch_start, mc.epoch_end):

        """Train."""
        model.train()
        train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}, Train', colour=mc.color_train)
        loss_train_history = []
        former_patient_image_feature = None
        former_patient_text_feature = None
        former_patient_label_survivals = None

        for i, patient_batch in enumerate(train_tqdm):
            """Data."""
            image3D = patient_batch['image3D'].to(mc.device)
            text = patient_batch['text'].to(mc.device)
            label_survivals = patient_batch['survivals'].to(mc.device)

            """Predict."""
            image_feature, text_feature, predicted_survivals_from_image, predicted_survivals_from_text \
                = model(image3D=image3D[0], text=text)
            image_feature = reduce(image_feature, 'b c -> c', 'mean')
            text_feature = reduce(text_feature, 'b c -> c', 'mean')

            """Loss & Optimize."""
            loss_survivals_of_image = criterion_MSE(predicted_survivals_from_image, label_survivals).to(mc.device)
            loss_survivals_of_text = criterion_MSE(predicted_survivals_from_text, label_survivals).to(mc.device)
            loss_similarity = 0
            if former_patient_image_feature is not None:
                loss_similarity = (
                        torch.dot(image_feature.detach(), former_patient_text_feature.detach()) +
                        torch.dot(former_patient_image_feature.detach(), text_feature.detach()) -
                        2 * torch.dot(image_feature.detach(), text_feature.detach()) +
                        2 * max(0.5, (label_survivals - former_patient_label_survivals).abs().item())).to(mc.device)
            loss = loss_survivals_of_image + loss_survivals_of_text + loss_similarity

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            train_tqdm.set_postfix(loss=f'{loss.item():.4f}')

            loss_train_history.append(np.array(loss.detach().cpu()))

            former_patient_image_feature = image_feature
            former_patient_text_feature = text_feature
            former_patient_label_survivals = label_survivals

        loss_train_history = np.array(loss_train_history)
        loss_train_history_mean = loss_train_history.mean(axis=0)

        summary_writer_train.add_scalar('MSE Loss', loss_train_history_mean, epoch + 1)

        lr_scheduler_model.step()

        """Test."""
        with torch.no_grad():
            model.eval()
            test_tqdm = tqdm(test_loader, desc=f'Epoch {epoch}, Test', colour=mc.color_test)

            label_survivals_history = []
            predicted_survivals_history = []
            loss_test_history = []

            for i, patient_batch in enumerate(test_tqdm):
                """Data."""
                image3D = patient_batch['image3D'].to(mc.device)
                text = patient_batch['text'].to(mc.device)
                label_survivals = patient_batch['survivals'].to(mc.device)

                """Predict."""
                image_feature, text_feature, predicted_survivals_from_image, predicted_survivals_from_text \
                    = model(image3D=image3D[0], text=text)

                """Loss."""
                loss_survivals_of_image = criterion_MSE(predicted_survivals_from_image, label_survivals)

                test_tqdm.set_postfix(loss_survivals_of_image=f'{loss_survivals_of_image.item():.4f}')

                label_survivals_array = np.array(label_survivals.squeeze(0).detach().cpu())
                predicted_survivals_array = np.array(predicted_survivals_from_image.squeeze(0).detach().cpu())
                loss_survivals_array = np.array(loss_survivals_of_image.detach().cpu())

                label_survivals_history.append(label_survivals_array)
                predicted_survivals_history.append(predicted_survivals_array)
                loss_test_history.append(loss_survivals_array)

            c_index = np.array([concordance_index(
                np.array(label_survivals_history)[:, i],
                np.array(predicted_survivals_history)[:, i]) for i in range(mc.survivals_len)]).mean(axis=0)

            loss_test_history_mean = np.array(loss_test_history).mean(axis=0)

            summary_writer_test.add_scalar('MSE Loss', loss_test_history_mean, epoch + 1)
            summary_writer_test.add_scalar('C Index', c_index, epoch + 1)

            """Save model."""
            if epoch >= mc.epoch_start_save_model - 1:
                if loss_test_history_mean < mc.min_test_loss:
                    mc.min_test_loss = loss_test_history_mean
                    """Remove former models."""
                    if len(glob.glob(mc.test_min_loss_model_path_reg)) > 0:
                        model_path = sorted(glob.glob(mc.test_min_loss_model_path_reg),
                                            key=lambda name: int(name.split('_')[-1].split('.')[0]))[-1]
                        if os.path.exists(model_path):
                            os.remove(model_path)
                    """Save the model that has had the min loss so far."""
                    if not os.path.exists(mc.model_path):
                        os.makedirs(mc.model_path)
                    torch.save(model.state_dict(), f'{mc.model_path}/test_min_loss_epoch_{epoch + 1}.pth')
                if (epoch - (mc.epoch_save_model_interval - 1)) % mc.epoch_save_model_interval == 0:
                    if not os.path.exists(mc.model_path):
                        os.makedirs(mc.model_path)
                    torch.save(model.state_dict(), f'{mc.model_path}/epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    train()

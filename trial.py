"""
This file is a completely independent lab.

Author: Han
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config as c
import mainConfig as mc
from myDataset import MyDataset


def trial():
    """
    Trial.
    """
    test_set = MyDataset(root=c.data_path, excel_path=c.excel_path, mode='test',
                         ki=0, k=c.k, transform=c.transforms_train, rand=True)
    test_loader = DataLoader(test_set, batch_size=mc.batch_size,
                             shuffle=True, num_workers=mc.num_workers)
    batch = next(iter(test_loader))
    label_survivals = torch.tensor([[1, 2, 3, 4]], dtype=torch.float)
    predicted_survivals = torch.tensor([[0, 1, 2, 3]], dtype=torch.float)
    criterion_MSE = nn.MSELoss()
    loss_survivals = criterion_MSE(predicted_survivals, label_survivals)
    print(loss_survivals)


def main():
    trial()


if __name__ == '__main__':
    main()

"""
This file is a completely independent lab.

Author: Han
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config as c
import mainConfig as mc
from myDataset import MyDataset


def trial():
    """
    Trial.
    """
    train_set = MyDataset(root=c.data_path, excel_path=c.excel_path, mode='train',
                          ki=4, k=c.k, transform=c.transforms_train, rand=True)
    data = train_set.__getitem__(0)
    print(data)


def main():
    trial()


if __name__ == '__main__':
    main()

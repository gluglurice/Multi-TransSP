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
    # train_set0 = MyDataset(root=c.data_path, excel_path=c.excel_path, mode='train',
    #                        ki=0, k=c.k, transform=c.transforms_train, rand=True)
    # train_set1 = MyDataset(root=c.data_path, excel_path=c.excel_path, mode='train',
    #                        ki=1, k=c.k, transform=c.transforms_train, rand=True)
    # print(train_set0.max_valid_slice_num, train_set1.max_valid_slice_num)
    a = np.array([1, 2, 3, 4, 5])
    print(a.mean(), a.std())


def main():
    trial()


if __name__ == '__main__':
    main()

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
from myDataset import MyDataset


def trial():
    """
    Trial.
    """
    test_set = MyDataset(root=c.data_path, excel_path=c.excel_path, mode='test',
                         ki=0, k=c.k, transform=c.transforms_train, rand=False)
    test_loader = DataLoader(test_set, batch_size=c.patient_batch_size,
                             shuffle=False, num_workers=c.num_workers)
    # data = next(iter(test_loader))
    # print(data['mha'], '\n')
    for i, batch in enumerate(test_loader):
        print(f'{i:03} batch     : {batch["mha"]}')
        batch_next = test_set.__getitem__((i+1) % len(test_set))
        print(f'{i:03} batch_next: {batch_next["mha"]}')


def main():
    trial()


if __name__ == '__main__':
    main()

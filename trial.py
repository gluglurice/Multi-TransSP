"""
This file is a completely independent lab.

Author: Han
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import reduce

import config as c
from myDataset import MyDataset


def trial():
    """
    Trial.
    """
    # test_set = MyDataset(root=c.data_path, excel_path=c.excel_path, mode='test',
    #                      ki=0, k=c.k, transform=c.transforms_train, rand=False)
    # test_loader = DataLoader(test_set, batch_size=c.patient_batch_size,
    #                          shuffle=False, num_workers=c.num_workers)
    # data = next(iter(test_loader))
    # print(data['mha'], '\n')
    a = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.float)
    b = torch.tensor([[2, 1, 4, 3]], dtype=torch.float)
    c = torch.zeros_like(a)
    for i in range(len(c)):
        for j in range(len(c[0])):
            c[i][j] = max(a[i][j], b[0][j])
    print(c)


def main():
    trial()


if __name__ == '__main__':
    main()

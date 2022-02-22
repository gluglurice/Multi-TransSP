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
    a = torch.tensor([[0]], dtype=torch.float)
    b = torch.tensor([[0.5]], dtype=torch.float)
    cos_similarity = torch.cosine_similarity(a, b, dim=-1)
    print(cos_similarity)


def main():
    trial()


if __name__ == '__main__':
    main()

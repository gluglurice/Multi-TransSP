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
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[3, 4, 5, 6]])
    l = [np.array(a), np.array(b)]
    l = np.array(l)
    avg_loss = l.mean(axis=0)
    print(avg_loss.squeeze())


def main():
    trial()


if __name__ == '__main__':
    main()

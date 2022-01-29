"""
Some variables

Author: Han
"""
import time

from torch import device, cuda
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


"""(1) Dataloader"""
excel_path = '../dataset/clinical.xlsx'
data_path = '../dataset/mha'
device = device('cuda' if cuda.is_available() else 'cpu')
transforms_train = transforms.Compose([
    transforms.Resize(int(332 * 1.12), InterpolationMode.BICUBIC),
    transforms.RandomCrop(332),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])
transforms_test = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])
k = 5                   # KFold k
num_workers = 1         # num_workers of data loader
text_length_dim = 2     # text is of torch.Size((1, 1, 12)) and we take the 2nd dimension as its length

"""(2) Network"""
model_resnet50_path = '../model/resnet50-19c8e357.pth'
model_path_reg = '../model/model_epoch_*.pth'
batch_size = 1
size = 332
lr = 0.02
epoch_start = 0
epoch_total = 20
epoch_decay = epoch_total // 2
conv_1_1_channel = 2048 + 12
sequence_length = 256
dimension = 11 * 11
survivals_len = 4

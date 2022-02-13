"""
Some variables

Author: Han
"""
from PIL import Image

from torch import device, cuda
from torchvision import transforms


"""(1) Dataloader"""
excel_path = '../dataset/clinical.xlsx'
data_path = '../dataset/mha'
device = device('cuda' if cuda.is_available() else 'cpu')
transforms_train = transforms.Compose([
    transforms.Resize(int(332 * 1.12), Image.BICUBIC),
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
model_resnet50_path = './model/resnet50-19c8e357.pth'
model_path_reg = './model/model_epoch_*.pth'
is_text = True
is_position = True
is_fastformer = True

sequence_length = 256
text_len = 12
survivals_len = 4

batch_size = 1
size = 332
lr = 0.002

epoch_start = 0
epoch_interval = 20
epoch_end = epoch_start + epoch_interval
epoch_total = epoch_interval * k
epoch_decay = 80

min_loss = 1e10

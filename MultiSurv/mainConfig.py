"""
Some variables

Author: Han
"""
from datetime import datetime

from PIL import Image

from torch import device, cuda
from torchvision import transforms


"""(1) Dataloader"""
size = 332
excel_path = '../dataset/clinical.xlsx'
data_path = '../dataset/mha'
device = device('cuda' if cuda.is_available() else 'cpu')
transforms_train = transforms.Compose([
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])
transforms_test = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])
k_start = 0             # KFold start ki
k = 5                   # KFold k
num_workers = 2         # num_workers of data loader
text_length_dim = 2     # text is of torch.Size((1, 1, 12)) and we take the 2nd num_patches as its length

"""(2) Network"""
patient_batch_size = 1
batch_size = 64
lr = 1e-3
weight_decay = 1e-6

date_time = datetime.now().strftime("%Y%m%d%H%M%S")
epoch_description = f'{date_time}_lr={lr}'
model_resnet_path = '../pretrainedModel/resnext50_32x4d-7cdf4587.pth'
model_path = f'./model/model_{epoch_description}'
model_path_reg = f'./model/model_{epoch_description}/*epoch_*.pth'
test_min_loss_model_path_reg = f'./model/model_{epoch_description}/test_min_loss_epoch_*.pth'
summary_path = f'./summary_{epoch_description}'

text_len = 12
survivals_len = 1

epoch_start = 0
epoch_end = 300
epoch_start_save_model = 150
epoch_save_model_interval = 10

min_train_loss = 1e10
min_test_loss = 1e10

color_train = '#f14461'
color_test = '#27ce82'

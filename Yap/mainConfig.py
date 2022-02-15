"""
Some variables

Author: Han
"""
from datetime import datetime

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
k_start = 0             # KFold start ki
k = 5                   # KFold k
num_workers = 1         # num_workers of data loader
text_length_dim = 2     # text is of torch.Size((1, 1, 12)) and we take the 2nd dimension as its length

"""(2) Network"""
is_text = True
batch_size = 1
size = 332
lr = 1e-3
weight_decay = 1e-6

date_time = datetime.now().strftime("%Y%m%d%H%M%S")
epoch_description = f'{date_time}_lr={lr}' \
                    f'{"_wo-text" if not is_text else ""}'
model_resnet50_path = '../pretrainedModel/resnet50-19c8e357.pth'
model_path = f'./model/model_{epoch_description}'
model_path_reg = f'./model/model_{epoch_description}/fold_*_epoch_*.pth'
summary_path = f'./summary_{epoch_description}'
test_model_path_reg = f'./model/model_20220212105551_lr=0.002/*'

sequence_length = 256
text_len = 12
survivals_len = 4

epoch_start = 0
epoch_end = 10
epoch_decay = 5

min_loss = 1e10

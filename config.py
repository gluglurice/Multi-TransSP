"""
Some variables

Author: Han
"""
from PIL import Image

from torch import device, cuda
from torchvision import transforms
from openpyxl.utils import column_index_from_string

"""Excel-related variables."""
excel_path = './dataset/clinical.xlsx'

col_id = column_index_from_string('B')

col_valid_layer = column_index_from_string('F')

col_feature_start = column_index_from_string('I')
col_feature_end = column_index_from_string('T')
col_label_start = column_index_from_string('U')
col_label_end = column_index_from_string('U')

"""Reading files."""
# input_path = './dataset/dicom'
# input_path_g = './dataset/dicom/chen dan-0006242445'
# data_path = './dataset/mha'
mha_files_path = '/*-cropped.mha'

"""Dataset"""

size = 332
device = device('cuda' if cuda.is_available() else 'cpu')
transforms_train = transforms.Compose([
    transforms.Resize(int(size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])
train_test_ratio = 5    # a number dividing the dataset by
k = 5                   # KFold k
seed = 2022             # seed of random.shuffle to guarantee the same dataset

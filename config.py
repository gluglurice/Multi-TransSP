"""
Some variables

Author: Han
"""
import time

from PIL import Image
from torch import device, cuda
from torchvision import transforms
from openpyxl.utils import column_index_from_string

"""Excel-related variables."""
excel_path = './dataset/clinical.xlsx'

col_id = column_index_from_string('B') - 1

col_min_layer = column_index_from_string('D') - 1
col_max_layer = column_index_from_string('E') - 1
col_total_layer = column_index_from_string('F') - 1

col_feature_start = column_index_from_string('I') - 1
col_feature_end = column_index_from_string('T') - 1
col_label_start = column_index_from_string('U') - 1
col_label_end = column_index_from_string('X') - 1

"""Reading files."""
input_path = './dataset/dicom'
input_path_g = './dataset/dicom/chen dan-0006242445'
data_path = './dataset/mha'
mha_files_path = '/*-cropped.mha'

"""Dataset"""
device = device('cuda' if cuda.is_available() else 'cpu')
transforms_train = transforms.Compose([
    transforms.Resize(int(332 * 1.12), Image.BICUBIC),
    transforms.RandomCrop(332),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])
train_test_ratio = 5    # a number dividing the dataset by
k = 5                   # KFold k
seed = 2022             # seed of random.shuffle to guarantee the same dataset

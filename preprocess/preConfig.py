"""
Some variables

Author: Han
"""
from openpyxl.utils import column_index_from_string

excel_path = '../dataset/clinical.xlsx'

col_id = column_index_from_string('B')

col_min_layer = column_index_from_string('D')
col_max_layer = column_index_from_string('E')
col_total_layer = column_index_from_string('F')

col_stage = column_index_from_string('U')
col_stage_T = column_index_from_string('V')
col_stage_N = column_index_from_string('W')

col_rt_days = column_index_from_string('M')
col_BMI = column_index_from_string('Q')

input_path = '../dataset/dicom'
output_path = '../dataset/mha'
input_type = 'cropped'
roi_list = ['GTVnx', 'PGTVnx', 'GTVnd', 'PGTVnd', 'CTV1', 'PCTV1', 'CTV2', 'PCTV2', 'CTVnd', 'PCTVnd']
# [z, (x1, y1), (x2, y2)]
roi_square = [[0, 0], [90, 90], [421, 421]]

process_num = 1

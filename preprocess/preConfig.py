"""
Some variables

Author: Han
"""
from openpyxl.utils import column_index_from_string

excel_path = '../dataset/clinical.xlsx'

col_id = column_index_from_string('B') - 1

col_min_layer = column_index_from_string('D') - 1
col_max_layer = column_index_from_string('E') - 1
col_total_layer = column_index_from_string('F') - 1

col_stage = column_index_from_string('U') - 1
col_stage_T = column_index_from_string('V') - 1
col_stage_N = column_index_from_string('W') - 1

col_rt_days = column_index_from_string('M') - 1
col_BMI = column_index_from_string('Q') - 1

input_path = '../dataset/dicom'
output_path = '../dataset/mha'
roi_list = ['GTVnx', 'PGTVnx', 'GTVnd', 'PGTVnd', 'CTV1', 'PCTV1', 'CTV2', 'PCTV2', 'CTVnd', 'PCTVnd']
# [z, (x1, y1), (x2, y2)]
roi_square = [[0, 0], [90, 90], [421, 421]]

process_num = 11

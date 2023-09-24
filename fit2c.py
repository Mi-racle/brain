import os.path

import numpy as np
import pandas as pd

from utils import ROOT

THERAPY_PATH = ROOT / 'data/表1-患者列表及临床信息.xlsx'
CURVE_PATH = ROOT / 'logs/2b/2bresult.xlsx'
OUTPUT_PATH = ROOT / 'logs/2c/2cresult.xlsx'


def normalize(arr):

    absolute_arr = np.abs(arr)
    max_element = np.max(absolute_arr)
    normalized_arr = arr / max_element

    return normalized_arr


table_therapy = pd.read_excel(
    THERAPY_PATH,
    sheet_name='患者信息',
    header=0,
    usecols=[16, 17, 18, 19, 20, 21, 22],
    nrows=100
)

inputs = table_therapy.to_numpy()

table_curve = pd.read_excel(
    CURVE_PATH,
    sheet_name='Sheet1',
    header=0,
    usecols=[1],
    nrows=100
)

target_as, target_bs = [], []

for i in range(100):

    params_str = table_curve.iloc[i, 0]
    params_str = params_str.replace('[', '').replace(']', '')
    param_str_list = params_str.split(' ')

    assert len(param_str_list) == 2

    target_as.append(float(param_str_list[0]))
    target_bs.append(float(param_str_list[1]))

target_as, target_bs = np.array(target_as), np.array(target_bs)

# a(Q, R, S, T, U, V, W) = alphas * [Q, R, S, T, U, V, W]^T
# b(Q, R, S, T, U, V, W) = betas * [Q, R, S, T, U, V, W]^T

alphas, _, _, _ = np.linalg.lstsq(inputs, target_as, rcond=None)
betas, _, _, _ = np.linalg.lstsq(inputs, target_bs, rcond=None)

normalized_alphas = normalize(alphas)
normalized_betas = normalize(betas)

print('Normalized alphas: ', normalized_alphas)
print('Normalized betas: ', normalized_betas)

output_table = pd.DataFrame()
output_table['normalized_alpha'] = normalized_alphas
output_table['normalized_beta'] = normalized_betas

if not os.path.exists(OUTPUT_PATH.parent):
    os.mkdir(OUTPUT_PATH.parent)

output_table.to_excel(OUTPUT_PATH, index=False)

import argparse
import os.path

import numpy as np
import pandas as pd

from utils import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='ED', type=str, help='Hemo or ED')
opt = parser.parse_args()

MODE = opt.mode

THERAPY_PATH = ROOT / 'data/表1-患者列表及临床信息.xlsx'
CURVE_PATH = ROOT / f'logs/2b/2b{MODE}result.xlsx'
OUTPUT_PATH = ROOT / f'logs/2c/2c{MODE}result.xlsx'


def normalize(arr):

    positive_arr = arr[arr > 0]
    summation = np.sum(positive_arr)
    normalized_arr = arr / summation

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

    if len(param_str_list) != 2:

        temp = []

        for ps in param_str_list:

            if ps != '':

                temp.append(ps)

        param_str_list = temp

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

import argparse
import os.path
import re

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

from utils import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='ED', type=str, help='Hemo or ED')
opt = parser.parse_args()

MODE = opt.mode

DATA_DIR_PATH = ROOT / 'data'
OUTPUT_PARAM_PATH = ROOT / f'logs/2a/2a{MODE}params.txt'
OUTPUT_XLSX_PATH = ROOT / f'logs/2a/2a{MODE}result.xlsx'
OUTPUT_IMG_PATH = ROOT / f'logs/2a/2a{MODE}.png'


# y = a * x^(1/2) * e^(-bx)
def func(x, _a, _b):

    return _a * x ** 0.5 * np.e ** (-_b * x)


def get_data(path):

    table = pd.read_excel(
        path,
        sheet_name='Sheet1',
        header=0,
        usecols=[0, 2, 3],
    )

    sub_ids = []
    intervals = []
    volumes = []

    last_interval = -1.
    adjusted_interval = 0.

    for i, row in table.iterrows():

        sub_id = table.iloc[i, 0]
        interval = table.iloc[i, 1]
        volume = table.iloc[i, 2]

        sub_ids.append(sub_id)

        if interval - last_interval < 1e-6:

            adjusted_interval += 1e-5
            intervals.append(adjusted_interval)

        else:

            last_interval = interval
            adjusted_interval = interval
            intervals.append(interval)

        volumes.append(volume)

    return np.array(sub_ids), np.array(intervals), np.array(volumes)


def extract_numbers_from_end(sub_id):

    match = re.search(r'\d+$', sub_id)

    if match:

        return int(match.group(0))

    else:

        raise 'Error'


if __name__ == '__main__':

    sub_ids, x_data, y_data = get_data(DATA_DIR_PATH / f'2a{MODE}.xlsx')

    # fit
    params, *covariance = curve_fit(f=func, xdata=x_data, ydata=y_data, bounds=([0, 0], [np.inf, np.inf]))

    print(f"params: {params}")

    with open(OUTPUT_PARAM_PATH, 'w') as fout:
        for param in params:
            fout.write(str(param) + '\n')
        fout.close()

    plt.scatter(x_data, y_data, s=10, label="data")
    plt.plot(x_data, func(x_data, *params), color='red', label="curve")
    plt.xlabel("time")
    plt.ylabel("volume")
    plt.legend()

    if not os.path.exists(OUTPUT_XLSX_PATH.parents[1]):
        os.mkdir(OUTPUT_XLSX_PATH.parents[1])

    if not os.path.exists(OUTPUT_XLSX_PATH.parents[0]):
        os.mkdir(OUTPUT_XLSX_PATH.parents[0])

    plt.savefig(OUTPUT_IMG_PATH)
    plt.clf()

    residuals = np.zeros(100)

    for i in range(len(sub_ids)):

        sub_id = extract_numbers_from_end(sub_ids[i])

        x, y = x_data[i], y_data[i]

        pred_y = func(x, *params)
        residual = abs(pred_y - y)

        residuals[sub_id - 1] += residual

    output_table = pd.DataFrame()
    output_table['残差'] = residuals
    output_table.to_excel(OUTPUT_XLSX_PATH, index=False)

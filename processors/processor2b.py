import argparse

import pandas as pd

from utils import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='ED', type=str, help='Hemo or ED')
parser.add_argument('-q', default='2b', type=str, help='2b or 3b')
opt = parser.parse_args()

MODE = opt.mode
question = opt.q

DATA_DIR_PATH = ROOT / 'data'
OUTPUT_PATH = DATA_DIR_PATH / f'{question}{MODE}.xlsx'

HEMO_VOLUME_COLS = [2, 25, 48, 71, 94, 117, 140, 163, 186]
ED_VOLUME_COLS = [13, 36, 59, 82, 105, 128, 151, 174, 197]

table_first = pd.read_excel(
    DATA_DIR_PATH / '表1-患者列表及临床信息.xlsx',
    sheet_name='患者信息',
    header=0,
    usecols=[0, 14],
)
table_first.drop(range(100, 160 if question == '2b' else 130), inplace=True)

table_serial = pd.read_excel(
    DATA_DIR_PATH / '表2-患者影像信息血肿及水肿的体积及位置.xlsx',
    sheet_name='Data',
    header=0,
    usecols=[1, 24, 47, 70, 93, 116, 139, 162, 185],
)
table_serial.drop(range(100, 160 if question == '2b' else 130), inplace=True)

table_volume = pd.read_excel(
    DATA_DIR_PATH / '表2-患者影像信息血肿及水肿的体积及位置.xlsx',
    sheet_name='Data',
    header=0,
    usecols=HEMO_VOLUME_COLS if MODE == 'Hemo' else ED_VOLUME_COLS,
)
table_volume.drop(range(100, 160 if question == '2b' else 130), inplace=True)

table_timestamp = pd.read_excel(
    DATA_DIR_PATH / '附表1-检索表格-流水号vs时间.xlsx',
    sheet_name='Sheet1',
    header=0,
    usecols=[2, 4, 6, 8, 10, 12, 14, 16, 18],
)
table_timestamp.drop(range(100, 160 if question == '2b' else 130), inplace=True)

# [{sub_id, serial, interval, volume}]
samples = []
header = ['sub_id']

for i in range(table_serial.shape[0]):

    row = table_serial.iloc[i]
    sub_id = str(table_first.iloc[i, 0])
    first_interval = float(table_first.iloc[i, 1])
    first_timestamp = table_timestamp.iloc[i, 0]

    sample = [sub_id]

    for j, serial in enumerate(row):

        # nan
        if serial != table_serial.iloc[i][j]:
            break

        if ('serial' + str(j)) not in header:
            header.append('serial' + str(j))

        if ('interval' + str(j)) not in header:
            header.append('interval' + str(j))

        if ('volume' + str(j)) not in header:
            header.append('volume' + str(j))

        serial = str(int(serial))
        volume = float(table_volume.iloc[i, j])
        timestamp = table_timestamp.iloc[i, j]

        time_delta = timestamp - first_timestamp
        interval = time_delta.total_seconds() / 3600. + first_interval

        sample.append(serial)
        sample.append(interval)
        sample.append(volume)

    samples.append(sample)

output_table = pd.DataFrame(samples, columns=header)
output_table.to_excel(OUTPUT_PATH, index=False)

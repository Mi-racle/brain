import pandas as pd

from utils import ROOT

DATA_PATH = ROOT / 'data'

table_first = pd.read_excel(
    DATA_PATH / '表1-患者列表及临床信息.xlsx',
    sheet_name='患者信息',
    header=0,
    usecols=[0, 14],
    nrows=100
)

table_serial = pd.read_excel(
    DATA_PATH / '表2-患者影像信息血肿及水肿的体积及位置.xlsx',
    sheet_name='Data',
    header=0,
    usecols=[1, 24, 47, 70, 93, 116, 139, 162, 185],
    nrows=100
)

table_volume = pd.read_excel(
    DATA_PATH / '表2-患者影像信息血肿及水肿的体积及位置.xlsx',
    sheet_name='Data',
    header=0,
    usecols=[13, 36, 59, 82, 105, 128, 151, 174, 197],
    nrows=100
)

table_timestamp = pd.read_excel(
    DATA_PATH / '附表1-检索表格-流水号vs时间.xlsx',
    sheet_name='Sheet1',
    header=0,
    usecols=[2, 4, 6, 8, 10, 12, 14, 16, 18],
    nrows=100
)

# [{sub_id, serial, interval, volume}]
samples = []

for i, row in table_serial.iterrows():

    sub_id = str(table_first.iloc[i, 0])
    first_interval = float(table_first.iloc[i, 1])
    first_timestamp = table_timestamp.iloc[i, 0]

    for j, serial in enumerate(row):

        # nan
        if serial != table_serial.iloc[i][j]:
            break

        serial = str(int(serial))
        volume = float(table_volume.iloc[i, j])
        timestamp = table_timestamp.iloc[i, j]

        time_delta = timestamp - first_timestamp
        interval = time_delta.total_seconds() / 3600.

        sample = {
            'sub_id': sub_id,
            'serial': serial,
            'interval': interval,
            'volume': volume
        }

        samples.append(sample)

samples.sort(key=lambda x: x['interval'])

keys = list(samples[0].keys())
output_table = pd.DataFrame([[sample[key] for key in keys] for sample in samples], columns=keys)
output_table.to_excel(DATA_PATH / '2a.xlsx', index=False)

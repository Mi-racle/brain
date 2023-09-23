import pandas as pd

pin = pd.read_excel('data/表1-患者列表及临床信息.xlsx', sheet_name='患者信息', header=0)
print(pin.head())

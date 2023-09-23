import pandas as pd

from utils import ROOT

DATA_PATH = ROOT / 'data'

table1 = pd.read_excel(
    DATA_PATH / '表1-患者列表及临床信息.xlsx',
    sheet_name='患者信息',
    header=0
)
assert '入院首次影像检查流水号' in table1.columns

gender = table1['性别']
new_gender = []
for g in gender:
    new_gender.append(0 if g == '男' else 1)
table1['性别'] = new_gender

pressure = table1['血压']
pressure1, pressure2 = [], []
for p in pressure:
    p = p.split('/')
    pressure1.append(int(p[0]))
    pressure2.append(int(p[1]))
table1['血压1'] = pressure1
table1['血压2'] = pressure2
table1.drop(['血压'], axis=1, inplace=True)

table1 = table1.iloc[:, 3:]

table2 = pd.read_excel(
    DATA_PATH / '表2-患者影像信息血肿及水肿的体积及位置.xlsx',
    sheet_name='Data',
    header=0
)
assert '首次检查流水号' in table2.columns
table2 = table2.iloc[:, :24]
table2.rename(columns={'首次检查流水号': '入院首次影像检查流水号'}, inplace=True)

table3 = pd.read_excel(
    DATA_PATH / '表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx',
    sheet_name='Hemo',
    header=0
)
assert '流水号' in table3.columns
table3.rename(columns={'流水号': '入院首次影像检查流水号'}, inplace=True)
table3 = table3.iloc[:, 1:]

final_table = pd.merge(table1, table2, on=['入院首次影像检查流水号'])
final_table = pd.merge(final_table, table3, on=['入院首次影像检查流水号'])

serials = final_table['入院首次影像检查流水号']
ids = final_table['ID']
final_table.drop(['入院首次影像检查流水号'], axis=1, inplace=True)
final_table.drop(['ID'], axis=1, inplace=True)

final_table = final_table.apply(lambda x: ((x - x.mean()) / x.std()) if x.std() != 0 else 1., axis=0)
# table1 = (table1 - table1.min()) / (table1.max() - table1.min())

final_table.insert(0, '入院首次影像检查流水号', serials)
final_table['入院首次影像检查流水号'] = final_table['入院首次影像检查流水号'].astype(str)
final_table.insert(0, 'ID', ids)

final_table.to_excel(DATA_PATH / '1b.xlsx', index=False)

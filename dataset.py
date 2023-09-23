import json
from pathlib import Path
from typing import Union, Sequence

import torch
import pandas as pd
from torch.utils.data import Dataset


class HematomaEnlargementDataset(Dataset):

    def __init__(self, data_path: Union[str, Path], rang: list[int], mode: str):

        super().__init__()

        table = pd.read_excel(
            Path(data_path) / '1b.xlsx',
            sheet_name='Sheet1',
            header=0
        )
        table = table.iloc[rang[0]:rang[1], :]

        self.ids = table['ID'].values
        self.data = torch.tensor(table.iloc[:, 2:].values, dtype=torch.float32)

        self.mode = mode

        if mode != 'train':

            return

        table_anno = pd.read_excel(
            Path(data_path) / 'e1_a_Process_new2.xlsx',
            sheet_name='Sheet3',
            header=1,
            usecols=[2]
        )
        # table_anno.replace(float('nan'), 48., inplace=True)
        # table_anno['血肿扩张时间'] /= 48.
        enlargement = table_anno['是否发生血肿扩张']

        no_enlargement = []
        for e in enlargement:
            no_enlargement.append(1 - e)

        table_anno['不扩张'] = no_enlargement

        self.labels = torch.tensor(table_anno.values, dtype=torch.float32)

        assert self.data.size(0) == self.labels.size(0)

    def __getitem__(self, index):

        if self.mode == 'train':
            # str, tensor(73), tensor(2)
            return self.ids[index], self.data[index], self.labels[index]

        else:

            return self.ids[index], self.data[index]

    def __len__(self):

        return self.data.size(0)

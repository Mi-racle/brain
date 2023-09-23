import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HematomaEnlargementDataset
from models.net import Net
from utils import ROOT, increment_path


def detect(
        model: nn.Module,
        loaded_set: DataLoader,
        output_dir
):

    output = pd.DataFrame()

    out_ids = []
    out_preds = []

    for i, (ids, inputs) in tqdm(enumerate(loaded_set), desc='Detect: ', total=len(loaded_set)):

        pred = model(inputs)

        out_ids.append(ids[0])
        out_preds.append(round(pred[0][0].item(), 4))

    output['ID'] = out_ids
    output['血肿扩张概率'] = out_preds

    output.to_excel(Path(output_dir) / '1bresult.xlsx', index=False)


def parse_opt(known=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', default=ROOT / 'logs' / 'train18' / 'best.pt')
    parser.add_argument('--data', default=ROOT / 'data')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--rang', default=[0, 160], type=int, nargs='+')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def run():

    opt = parse_opt()
    weights = opt.weights
    data_path = opt.data
    batch_size = opt.batch_size
    device = 'cpu' if not torch.cuda.is_available() or opt.device == 'cpu' else 'cuda:' + str(opt.device)
    device = torch.device(device)
    visualize = opt.visualize

    model = Net(73)
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.to(device)

    data = HematomaEnlargementDataset(data_path, [0, 160], 'test')
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)

    if not os.path.exists(ROOT / 'logs'):
        os.mkdir(ROOT / 'logs')
    output_dir = increment_path(ROOT / 'logs' / 'detect', mkdir=True)

    detect(model, loaded_set, output_dir)

    print(f'\033[92mResults saved to {output_dir}\033[0m')


if __name__ == '__main__':

    run()

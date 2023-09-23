import argparse
import os

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HematomaEnlargementDataset
from logger import Logger
from loss import Loss
from models.net import Net
from utils import increment_path, log_epoch, ROOT


def train(
        model: nn.Module,
        loaded_set: DataLoader,
        loss_computer: Loss,
        optimizer: Optimizer
):

    total_loss = 0

    for i, (ids, inputs, targets) in tqdm(enumerate(loaded_set), total=len(loaded_set)):

        pred = model(inputs)
        loss = loss_computer(pred, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loaded_set)

    return average_loss


def parse_opt(known=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default=ROOT / 'data', type=str)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='cpu or 0 (cuda)')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--early-stopping', default=20, type=int)
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--rang', default=[0, 160], type=int, nargs='+')
    parser.add_argument('--lr', default=1e-3, type=float)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def run():

    opt = parse_opt()
    data_path = opt.data
    batch_size = opt.batch_size
    epochs = opt.epochs
    early_stopping = opt.early_stopping
    lr = opt.lr

    model = Net(73)
    data = HematomaEnlargementDataset(data_path, [0, 100], 'train')
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)
    loss_computer = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    if not os.path.exists('logs'):
        os.mkdir('logs')
    output_dir = increment_path('logs/train')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger = Logger(output_dir)

    best_loss = float('inf')
    patience = early_stopping

    for epoch in range(0, epochs):

        print(f'Epoch {epoch}:')
        model.train()
        loss = train(model, loaded_set, loss_computer, optimizer)
        log_epoch(logger, epoch, model, loss, best_loss)

        if loss < best_loss:

            best_loss = loss
            patience = early_stopping

        else:

            patience -= 1

            if patience < 0:

                break

    print(f'\033[92mResults have been saved to {output_dir}\033[0m')


if __name__ == '__main__':

    run()

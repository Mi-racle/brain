import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, ch):
        torch.argmax()
        super().__init__()

        neu_num0 = 128
        neu_num1 = 64
        neu_num2 = 16

        self.linear0 = nn.Linear(ch, neu_num0)
        self.linear1 = nn.Linear(neu_num0, neu_num1)
        self.linear2 = nn.Linear(neu_num1, neu_num2)
        self.fc = nn.Linear(neu_num2, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x

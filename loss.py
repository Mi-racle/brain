from torch import nn


class Loss:
    def __init__(self):

        super(Loss, self).__init__()

        self.loss = nn.BCELoss()

    def __call__(self, pred, target):

        return self.loss(pred, target)

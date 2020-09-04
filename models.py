import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressor(nn.Module):
    def __init__(self, win_dim, in_dim, out_dim):
        super(LogisticRegressor, self).__init__()
        self.win_dim, self.in_dim = win_dim, in_dim

        self.main = nn.Sequential(
            nn.Linear(win_dim * in_dim, out_dim)
        )

    def forward(self, x):
        return self.main(x)

class Convolutional(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Convolutional, self).__init__()

        self.l1 = nn.Conv2d(4, 8, kernel_size=2, stride=1)
        self.l2 = nn.Conv2d(4, 8, kernel_size=2, stride=1)
        self.l3 = nn.Conv2d(8, 12, kernel_size=2, stride=1)
        self.classify = nn.Linear(32, out_dim)
    
    def forward(self, x):
        print(x.shape)
        y = self.l1(x)
        print(y.shape)
        # y = self.l2(x)
        # print(y.shape)
        # y = self.l3(x)
        y = y.view(-1, 32)
        return self.classify(y)
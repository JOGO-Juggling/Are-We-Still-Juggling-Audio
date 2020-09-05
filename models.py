import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressor(nn.Module):
    def __init__(self, in_dim, win_dim, out_dim):
        super(LogisticRegressor, self).__init__()
        
        self.l1 = nn.Linear(win_dim * in_dim, 60)
        # self.l2 = nn.Linear(60, 60)
        # self.l3 = nn.Linear(60, 60)
        self.l4 = nn.Linear(60, out_dim)

    def forward(self, x):
        y = x
        y = self.l1(y)
        # y = self.l2(y)
        # y = self.l3(y)
        y = self.l4(y)
        return y

class Convolutional(nn.Module):
    def __init__(self, in_dim, w_dim, out_dim):
        super(Convolutional, self).__init__()
        self._in_dim = in_dim

        self.l1 = nn.Conv2d(1, 2, kernel_size=w_dim, stride=1)
        self.l2 = nn.Linear(2 * (in_dim - 4), in_dim - 4)
        self.l3 = nn.Linear(in_dim - 4, in_dim // 2)
        self.l4 = nn.Linear(in_dim // 2, out_dim)
    
    def forward(self, x):
        y = x
        y = self.l1(y)
        y = y.view(-1, 2 * (self._in_dim - 4))
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        return y
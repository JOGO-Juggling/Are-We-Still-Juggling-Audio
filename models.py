import torch
import torch.nn as nn
import torch.nn.functional as F

N_F = 4

class LogisticRegressor(nn.Module):
    def __init__(self, in_dim, win_dim, out_dim):
        super(LogisticRegressor, self).__init__()
        
        self.l1 = nn.Linear(win_dim * in_dim, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 20)
        self.l4 = nn.Linear(20, out_dim)
        
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        y = x
        y = F.leaky_relu(self.l1(y), 0.2)
        y = F.leaky_relu(self.l2(y), 0.2)
        y = self.drop(y)
        y = F.leaky_relu(self.l3(y), 0.2)
        y = self.drop(y)
        y = F.leaky_relu(self.l4(y), 0.2)
        return y

class Convolutional(nn.Module):
    def __init__(self, in_dim, w_dim, out_dim):
        super(Convolutional, self).__init__()
        self._in_dim = in_dim
        self._w_dim = w_dim

        self.l1 = nn.Conv2d(1, N_F, kernel_size=w_dim, stride=1)
        self.l2 = nn.BatchNorm2d(N_F)

        self.l3 = nn.Linear(N_F * self._in_dim - N_F * (w_dim - 1), 2 * in_dim)
        self.l4 = nn.Linear(2 * in_dim, in_dim // 2)
        self.l5 = nn.Linear(in_dim // 2, out_dim)

        self.drop = nn.Dropout(0.2)
    
    def forward(self, x):
        y = x
        y = F.leaky_relu(self.l1(y), 0.2)
        y = self.l2(y)

        y = y.view(-1, N_F * self._in_dim - N_F * (self._w_dim - 1))
        y = F.leaky_relu(self.l3(y), 0.2)
        y = self.drop(y)
        y = F.leaky_relu(self.l4(y), 0.2)
        y = self.drop(y)
        y = F.leaky_relu(self.l5(y), 0.2)
        return y
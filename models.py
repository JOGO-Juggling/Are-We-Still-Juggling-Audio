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
        x = x.view(-1, self.win_dim * self.in_dim)
        return self.main(x)

class Convolutional(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Convolutional, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2)
            nn.Conv1d(4, 8, kernel_size=2, stride=1)
            nn.Conv1d(4, 8, kernel_size=2, stride=1)
        )

        self.classify = nn.Linear(32, out_dim)
    
    def forward(self, x):
        x = x.view(-1, self.win_dim * self.in_dim)
        y = self.main(x)
        # print(y.shape)
        y = y.view(-1, 32)
        return self.classify(y)
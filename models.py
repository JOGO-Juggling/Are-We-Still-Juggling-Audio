import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegressor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.main(x)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class BasicCondGaussian(nn.Module):
    def __init__(self, mean_scale, std):
        super().__init__()
        self.register_buffer("mean_scale", torch.tensor(mean_scale))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, y):
        return self.mean_scale*y, self.std.expand_as(y)


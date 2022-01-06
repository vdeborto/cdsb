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


class BasicRegressGaussian(nn.Module):
    def __init__(self, mean_module, mean_scale, std):
        super().__init__()
        self.add_module("mean_module", mean_module)
        self.register_buffer("mean_scale", torch.tensor(mean_scale))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, y):
        mean_x = self.mean_module(y)
        return self.mean_scale*mean_x, self.std.expand_as(mean_x)
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os


def data_distrib(npar, data):
    x = torch.randn(npar, 2)
    A = 0.4 + 0.4 * (1 + torch.erf(x[:, :1] / np.sqrt(2)))
    B = 0.01 + 0.15 * (1 + torch.erf(x[:, 1:] / np.sqrt(2)))
    normal = torch.randn(npar, 5) * np.sqrt(1e-3)

    y_mean = A * (1 - torch.exp(-B * torch.arange(1, 6)))
    y = y_mean + normal
    return x, y


def biochemical_ds(root, npar, data_tag):
    data_path = os.path.join(root, data_tag, "data.pt")
    if os.path.isfile(data_path):
        init_sample_x, init_sample_y = torch.load(data_path)
        assert init_sample_x.shape[0] == npar
        print("Loaded dataset biochemical", data_tag)
    else:
        os.makedirs(os.path.join(root, data_tag), exist_ok=True)
        init_sample_x, init_sample_y = data_distrib(npar, data_tag)
        torch.save([init_sample_x, init_sample_y], data_path)
        print("Created new dataset biochemical", data_tag)
    init_ds = TensorDataset(init_sample_x, init_sample_y)
    return init_ds

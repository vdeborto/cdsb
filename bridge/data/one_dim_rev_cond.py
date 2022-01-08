import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset
import os


def y_cond_x_sim(x, data, args):
    y_std = args.data.y_std
    if data == 'type1':
        y = x**2 + 1 + y_std * torch.randn_like(x)
    elif data == 'type4':
        y = x + y_std * torch.randn_like(x)
    return y


def data_distrib(npar, data, args, x=None):
    x_std = args.data.x_std
    if x is None:
        x = x_std * torch.randn(npar, 1)
    y = y_cond_x_sim(x, data, args)
    return x, y


def one_dim_rev_cond_ds(root, npar, data_tag, args, x=None, shuffle=False):
    data_path = os.path.join(root, data_tag, "data.pt")
    if args.load and os.path.isfile(data_path) and x is None:
        init_sample_x, init_sample_y = torch.load(data_path)
        assert init_sample_x.shape[0] == npar
        print("Loaded dataset 1d_rev_cond", data_tag)
    else:
        os.makedirs(os.path.join(root, data_tag), exist_ok=True)
        init_sample_x, init_sample_y = data_distrib(npar, data_tag, args, x=x)
        if x is None:
            torch.save([init_sample_x, init_sample_y], data_path)
        elif shuffle:
            init_sample_y = init_sample_y[torch.randperm(init_sample_y.shape[0])]
        print("Created new dataset 1d_rev_cond", data_tag)
    init_ds = TensorDataset(init_sample_x, init_sample_y)
    return init_ds

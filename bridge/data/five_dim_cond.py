import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset
import os


def data_distrib(npar, data):

    y = torch.randn(npar, 5)
    
    if data == 'type1':
        normal = torch.randn(npar, 1)
        x_mean = y[:, 0]**2 + torch.exp(y[:, 1] + y[:, 2]/3) + torch.sin(y[:, 3] + y[:, 4])
        x = x_mean.reshape(npar, 1) + normal

    elif data == 'type2':
        normal = torch.randn(npar, 1)
        x_mean = y[:, 0]**2 + torch.exp(y[:, 1] + y[:, 2]/3) + y[:, 3] - y[:, 4]
        x_std = 0.5 + y[:, 1]**2/2 + y[:, 4]**2/2
        x = x_mean.reshape(npar, 1) + normal * x_std.reshape(npar, 1)

    elif data == 'type3':
        uniform = torch.rand(npar, 1)
        normal_n = torch.randn(npar, 1) - 2
        normal_p = torch.randn(npar, 1) + 2
        eps = torch.where(uniform < 0.5, normal_n, normal_p)
        x = (5 + y[:, 0]**2/3 + y[:, 1]**2 + y[:, 2]**2 + y[:, 3] + y[:, 4]).reshape(npar, 1) * torch.exp(0.5*eps)

    elif data == 'type4':
        uniform = torch.rand(npar, 1)
        normal_n = torch.randn(npar, 1)*0.25 - y[:, 0:1]
        normal_p = torch.randn(npar, 1)*0.25 + y[:, 0:1]
        x = torch.where(uniform < 0.5, normal_n, normal_p)

    return x, y

def five_dim_cond_ds(root, npar, data_tag):
    data_path = os.path.join(root, data_tag, "data.pt")
    if os.path.isfile(data_path): 
        init_sample_x, init_sample_y = torch.load(data_path)
        assert init_sample_x.shape[0] == npar
        print("Loaded dataset 5d_cond", data_tag)
    else:
        os.makedirs(os.path.join(root, data_tag), exist_ok=True)
        init_sample_x, init_sample_y = data_distrib(npar, data_tag)
        torch.save([init_sample_x, init_sample_y], data_path)
        print("Created new dataset 5d_cond", data_tag)
    init_ds = TensorDataset(init_sample_x, init_sample_y)
    return init_ds

import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset

# checker/pinwheel/8gaussians can be found at 
# https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py#L8

def data_distrib(npar, data):

    y = 6. * torch.rand(npar, 1) - 3.
    tanh = torch.nn.Tanh()
    
    if data == 'type1':
        gamma_dis = torch.distributions.gamma.Gamma(1, 1/0.3)
        gamma = gamma_dis.sample_n(npar).reshape(npar, 1)
        x = tanh(y) + gamma

    if data == 'type2':
        std = torch.sqrt(torch.Tensor([0.05]))
        gamma = std * torch.randn(npar, 1)
        x = tanh(y + gamma)

    if data == 'type3':
        gamma_dis = torch.distributions.gamma.Gamma(1, 1/0.3)
        gamma = gamma_dis.sample_n(npar).reshape(npar, 1)
        x = gamma * tanh(y)

    init_sample_x = x
    init_sample_y = y
        
    init_sample_x = init_sample_x.float()
    init_sample_y = init_sample_y.float()

    return init_sample_x, init_sample_y

def one_dim_cond_ds(npar, data_tag):
    init_sample_x, init_sample_y = data_distrib(npar, data_tag)
    init_ds = TensorDataset(init_sample_x, init_sample_y)
    return init_ds

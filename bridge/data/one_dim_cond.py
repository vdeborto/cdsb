import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset

# checker/pinwheel/8gaussians can be found at 
# https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py#L8

def data_distrib(npar, data):

    y = 6. * torch.rand(npar, 1) - 3.
    tanh = torch.nn.Tanh()
    init_sample = torch.zeros((npar, 1, 2))
    
    if data == 'type1':
        gamma_dis = torch.distributions.gamma.Gamma(0.3, 1)                
        gamma = gamma_dis.sample_n(npar).reshape(npar, 1)
        x = tanh(y) + gamma

    if data == 'type2':
        std = torch.sqrt(torch.Tensor([0.05]))
        gamma = std * torch.randn(npar, 1)
        x = tanh(y + gamma)

    if data == 'type3':
        gamma_dis = torch.distributions.gamma.Gamma(0.3, 1)                
        gamma = gamma_dis.sample_n(npar).reshape(npar, 1)
        x = gamma * tanh(y)

    init_sample[...,0] = x
    init_sample[...,1] = y
        
    init_sample = init_sample.float()

    return init_sample

def one_dim_cond_ds(npar, data_tag):
    init_sample = data_distrib(npar, data_tag)
    init_ds = TensorDataset(init_sample)
    return init_ds

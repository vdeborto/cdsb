import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset

# checker/pinwheel/8gaussians can be found at 
# https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py#L8

def data_distrib(npar, data):

    x = 6. * torch.rand(npar, 1) - 3.
    tanh = torch.nn.Tanh()
    init_sample = torch.zeros((2, npar, 1))
    
    if data == 'type1':
        gamma_dis = torch.distributions.gamma.Gamma(1, 0.3)                
        gamma = gamma_dis.sample_n(npar).reshape(npar, 1)
        y = tanh(x) + gamma

    if data == 'type2':
        gamma = torch.randn(npar, 1) * torch.sqrt(0.05)
        y = tanh(x + gamma)

    if data == 'type3':
        gamma_dis = torch.distributions.gamma.Gamma(1, 0.3)                
        gamma = gamma_dis.sample_n(npar).reshape(npar, 1)
        y = gamma * tanh(x)

    init_sample[0] = x
    init_sample[1] = y
        
    init_sample = init_sample.float()

    return init_sample

def one_dim_cond_ds(npar, data_tag):
    init_sample = data_distrib(npar, data_tag)
    init_ds = TensorDataset(init_sample)
    return init_ds

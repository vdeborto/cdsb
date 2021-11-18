import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset
import os

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

def one_dim_cond_ds(root, npar, data_tag):
    data_path = os.path.join(root, data_tag, "data.pt")
    if os.path.isfile(data_path): 
        init_sample_x, init_sample_y = torch.load(data_path)
        assert init_sample_x.shape[0] == npar
        print("Loaded dataset 1d_cond", data_tag)
    else:
        os.makedirs(os.path.join(root, data_tag), exist_ok=True)
        init_sample_x, init_sample_y = data_distrib(npar, data_tag)
        torch.save([init_sample_x, init_sample_y], data_path)
        print("Created new dataset 1d_cond", data_tag)
    init_ds = TensorDataset(init_sample_x, init_sample_y)
    return init_ds

if __name__ == "__main__":
    data = torch.cat(one_dim_cond_ds(10000, "type3").tensors, 1)
    # import seaborn as sns
    # sns.kdeplot(x=data[:, 1], y=data[:, 0])
    # plt.xlim(-4, 4)
    # plt.ylim(-1.5, 2.5)
    # plt.show()
    from scipy import stats
    import matplotlib.pyplot as plt
    density = lambda xy: stats.gaussian_kde(data.flip(1).T)(xy.T)
    # density = lambda xy: 1/6 * stats.gamma.pdf(xy[:, 1] / np.tanh(xy[:, 0]), 1, scale=0.3) * np.where(np.abs(xy[:, 0]) < 3, 1, 0)
    delta = 0.025
    x = np.arange(-3.5, 3.5, delta)
    y = np.arange(-0.6, 0.6, delta)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, density(np.stack([X, Y], 2).reshape((-1, 2))).reshape(X.shape), levels=8)
    plt.show()
import numpy as np
from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .poly import PolynomialRegressor


def l2_distance(FX, FY, stable=True):
    if stable:
        FK = torch.sum((FX[:, None, :] - FY[None, :, :]) ** 2, -1)
    else:
        FK = (FX ** 2).sum(-1, keepdim=True)
        FK = FK + (FY ** 2).sum(-1, keepdim=True).t()
        FK -= 2 * (FX[:, None, :] * FY[None, :, :]).sum(-1)
    return FK


def estimate_median_distance(data):
    return np.median(pdist(data.detach().cpu().numpy()))


class MaternKernel(nn.Module):
    def __init__(self, nets=[torch.nn.Identity()], sigma=1.0, lam=0.01, p=np.inf, train_sigma=True, train_lam=True):
        super().__init__()
        self.train_sigma = train_sigma
        self.train_lam = train_lam

        self.log_lam = nn.Parameter(torch.tensor(lam).log(), requires_grad=train_lam)
        self.log_sigma = nn.Parameter(torch.tensor(sigma).log(), requires_grad=train_sigma)
        self.kernel_networks = nn.ModuleList(nets)
        self.p = p

    def forward(self, x, y):
        return self.gram(x, y)

    def dgg(self, X, Y, g):
        FX = g(X)
        FY = g(Y)
        FK = l2_distance(FX, FY)
        return FK

    def gram(self, X, Y):
        G = 0
        for k in self.kernel_networks:
            G = G + self.dgg(X, Y, k)

        if self.p == np.inf:
            G = (-G / (2 * self.log_sigma.exp() ** 2)).exp()
            return G
        else:
            distance = G.sqrt() / self.log_sigma.exp()
            exp_component = torch.exp(-np.sqrt(self.p * 2) * distance)

            if self.p == 0.5:
                constant_component = 1
            elif self.p == 1.5:
                constant_component = (np.sqrt(3) * distance).add(1)
            elif self.p == 2.5:
                constant_component = (np.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component


class KernelRidgeRegressor(nn.Module):
    def __init__(self, x_dim, y_dim, kernel_fn, num_steps, train_iter=30, lr=0.005):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_steps = num_steps
        self.train_iter = train_iter
        self.lr = lr

        self.kernels = nn.ModuleList([kernel_fn() for _ in range(num_steps)])
        self.Ks = []
        self.xs = []
        self.ys = []
        self.outs = []

        self.linear_model = PolynomialRegressor(x_dim, y_dim, 1, 1, num_steps)

    def fit(self, x, y, out, t):
        self.linear_model.fit(x, y, out, t)

        for k in range(self.num_steps):
            t_idx = torch.where(t[:, 0] == k)[0]

            x_train = x[t_idx]
            y_train = y[t_idx]
            out_train = out[t_idx]

            linear_out_train = self.linear_model(x_train, y_train, torch.tensor([[k]]))
            out_train = out_train - linear_out_train

            if self.kernels[k].train_sigma or self.kernels[k].train_lam:
                t_len = x_train.shape[0]
                x_train, x_valid = x_train.split([t_len - t_len//5, t_len//5])
                out_train, out_valid = out_train.split([t_len - t_len//5, t_len//5])

            self.xs.append(x_train)
            self.outs.append(out_train)

            self.kernels[k].log_sigma.data = torch.tensor(np.log(estimate_median_distance(x_train))).float()

            if self.kernels[k].train_sigma or self.kernels[k].train_lam:
                optimizer = torch.optim.Adam(self.kernels[k].parameters(), lr=self.lr)
                for i in range(self.train_iter):
                    optimizer.zero_grad()
                    loss = functional.mse_loss(self.forward_krr(x_valid, None, torch.tensor([[k]])), out_valid)
                    loss.backward()
                    optimizer.step()

            # self.Ks.append(self.compute_kernel(x_train, k))

    def forward_krr(self, x, y, t):
        k = t[0, 0]
        assert torch.all(t == k)

        # similarity between sleep and wake data
        G = self.kernels[k](self.xs[k], x)
        if len(self.Ks) > k:
            K = self.Ks[k]
        else:
            K = self.compute_kernel(self.xs[k], k)

        # this computes G.t() @ K^{-1}
        GKinv = torch.linalg.solve(K, G).t()

        return GKinv @ self.outs[k]

    def forward(self, x, y, t):
        linear_out = self.linear_model(x, y, t)
        krr_out = self.forward_krr(x, y, t)
        return linear_out + krr_out


    def compute_kernel(self, x, k):
        return self.kernels[k](x, x) + self.kernels[k].log_lam.exp() * torch.eye(x.shape[0], device=x.device)


import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, RidgeCV


class DimwiseBasisRegressor(nn.Module):
    def __init__(self, x_dim, y_dim, deg, basis, num_steps, x_radius=None, y_radius=None, alphas=[1e-6, 1e-4, 1e-2, 1e-1, 1.]):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_steps = num_steps
        self.deg = deg
        # self.models = [LinearRegression() for _ in range(self.num_steps*self.x_dim)]
        self.models = [RidgeCV(alphas=alphas) for _ in range(self.num_steps*self.x_dim)]

        self.basis = basis
        self.x_radius = x_radius
        self.y_radius = y_radius

        if self.basis == 'rbf':
            self.basis_locs = [None for _ in range(self.num_steps*self.x_dim)]
            self.basis_scales = [None for _ in range(self.num_steps*self.x_dim)]

        self.X_means = [None for _ in range(self.num_steps*self.x_dim)]
        self.X_stds = [None for _ in range(self.num_steps*self.x_dim)]

    def register_basis(self, X, i):
        if self.basis == 'rbf':
            if self.deg >= 2:
                gamma = 2
                self.basis_locs[i] = torch.quantile(X, torch.arange(1, self.deg)/self.deg, dim=0)
                if self.deg == 2:
                    qq = torch.quantile(X, torch.tensor([0.25, 0.75]), dim=0)
                    self.basis_scales[i] = (qq[1:] - qq[:1]) / 2 * gamma
                elif self.deg == 3:
                    self.basis_scales[i] = (self.basis_locs[i][np.ones(2)] - self.basis_locs[i][np.zeros(2)]) * gamma
                else:
                    self.basis_scales[i] = (self.basis_locs[i][torch.tensor([*range(1, self.deg - 1), -1])] -
                                            self.basis_locs[i][torch.tensor([0, *range(self.deg - 2)])]) / 2 * gamma

    def compute_basis(self, X, i):
        if self.basis == 'rbf':
            if self.deg >= 2:
                basis = torch.exp(-0.5 * ((X.unsqueeze(1) - self.basis_locs[i]) / self.basis_scales[i]) ** 2)
                basis = basis.reshape(X.shape[0], -1)
                X = torch.cat([X, basis], dim=-1)

        return X

    def get_x_radius_index(self, d):
        if self.x_radius is None:
            return np.arange(self.x_dim)
        else:
            idx = np.arange(d - self.x_radius + 1, d + self.x_radius)
            return idx % self.x_dim

    def get_y_radius_index(self, d):
        if self.y_radius is None:
            return np.arange(self.y_dim)
        else:
            assert self.x_dim == self.y_dim
            idx = np.arange(d - self.y_radius + 1, d + self.y_radius)
            return idx % self.y_dim

    def forward(self, x, y, t):
        k = t[0, 0]
        assert torch.all(t == k)
        out = []

        for d in range(self.x_dim):
            i = k*self.x_dim+d

            x_input_index = self.get_x_radius_index(d)
            y_input_index = self.get_y_radius_index(d)
            x_input = x[:, x_input_index]
            y_input = y[:, y_input_index]

            X = torch.cat([x_input, y_input], dim=-1)
            X = self.compute_basis(X, i)

            X_preprocessed = (X - self.X_means[i]) / self.X_stds[i]

            out.append(torch.from_numpy(self.models[i].predict(X_preprocessed)))
        out = torch.cat(out, dim=-1).float()
        return out

    def fit(self, x, y, out, t):
        for k in range(self.num_steps):
            t_idx = torch.where(t[:, 0] == k)[0]

            for d in range(self.x_dim):
                i = k*self.x_dim+d

                x_input_index = self.get_x_radius_index(d)
                y_input_index = self.get_y_radius_index(d)
                x_input = x[t_idx, x_input_index]
                y_input = y[t_idx, y_input_index]

                X = torch.cat([x_input, y_input], dim=-1)
                self.register_basis(X, i)

                X = self.compute_basis(X, i)

                self.X_means[i] = X.mean(0)
                self.X_stds[i] = X.std(0)

                X_preprocessed = (X - self.X_means[i]) / self.X_stds[i]

                self.models[i].fit(X_preprocessed, out[t_idx, d].unsqueeze(1))


# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#     x = torch.randn(200, 1)
#     y = torch.randn(200, 1)
#     out = torch.exp(-x**2/2)
#
#     regressor = DimwiseBasisRegressor(1, 1, 1, 2)
#     regressor.fit(x, y, out, torch.zeros(200, 1).to(int))
#
#     x_test = torch.linspace(-2, 2, 200).unsqueeze(1)
#     out_test = regressor(x_test, torch.randn(200, 1), torch.zeros(200, 1).to(int))
#
#     plt.plot(x_test, out_test)
#     plt.plot(x_test, torch.exp(-x_test**2/2))
#     plt.show()

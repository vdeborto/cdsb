import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, RidgeCV


class PolynomialRegressor(nn.Module):
    def __init__(self, x_dim, y_dim, x_deg, y_deg, num_steps):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_deg = x_deg
        self.y_deg = y_deg
        self.num_steps = num_steps
        # self.models = [LinearRegression() for _ in range(self.num_steps)]
        self.models = [RidgeCV(alphas=[1e-6, 1e-4, 1e-2, 1e-1, 1.]) for _ in range(self.num_steps)]

        self.x_means = []
        self.x_stds = []
        self.y_means = []
        self.y_stds = []

    def compute_basis(self, x, y):
        X = x.clone()
        for d in range(2, self.x_deg+1):
            X = torch.cat([X, x**d], dim=-1)

        X = torch.cat([X, y], dim=-1)
        for d in range(2, self.y_deg+1):
            X = torch.cat([X, y**d], dim=-1)

        return X

    def forward(self, x, y, t):
        k = t[0, 0]
        assert torch.all(t == k)

        x_preprocessed = (x - self.x_means[k]) / self.x_stds[k]
        y_preprocessed = (y - self.y_means[k]) / self.y_stds[k]

        X = self.compute_basis(x_preprocessed, y_preprocessed)
        out = torch.from_numpy(self.models[k].predict(X))
        return out.float()

    def fit(self, x, y, out, t):
        for k in range(self.num_steps):
            t_idx = torch.where(t[:, 0] == k)[0]

            self.x_means.append(x[t_idx].mean(0))
            self.x_stds.append(x[t_idx].std(0))
            self.y_means.append(y[t_idx].mean(0))
            self.y_stds.append(y[t_idx].std(0))

            x_preprocessed = (x[t_idx] - self.x_means[k]) / self.x_stds[k]
            y_preprocessed = (y[t_idx] - self.y_means[k]) / self.y_stds[k]

            X = self.compute_basis(x_preprocessed, y_preprocessed)
            self.models[k].fit(X, out[t_idx])


class DimwisePolynomialRegressor(nn.Module):
    def __init__(self, x_dim, y_dim, x_deg, y_deg, num_steps, x_dimwise=False, y_dimwise=False):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_deg = x_deg
        self.y_deg = y_deg
        self.num_steps = num_steps
        # self.models = [LinearRegression() for _ in range(self.num_steps*self.x_dim)]
        self.models = [RidgeCV(alphas=[1e-6, 1e-4, 1e-2, 1e-1, 1.]) for _ in range(self.num_steps*self.x_dim)]

        self.x_dimwise = x_dimwise
        self.y_dimwise = y_dimwise

        self.x_means = []
        self.x_stds = []
        self.y_means = []
        self.y_stds = []

    def compute_basis(self, x, y):
        X = x.clone()
        for d in range(2, self.x_deg+1):
            X = torch.cat([X, x**d], dim=-1)

        X = torch.cat([X, y], dim=-1)
        for d in range(2, self.y_deg+1):
            X = torch.cat([X, y**d], dim=-1)

        return X

    def forward(self, x, y, t):
        k = t[0, 0]
        assert torch.all(t == k)
        out = []

        x_preprocessed = (x - self.x_means[k]) / self.x_stds[k]
        y_preprocessed = (y - self.y_means[k]) / self.y_stds[k]

        for d in range(self.x_dim):
            if self.x_dimwise:
                x_input = x_preprocessed[:, d].unsqueeze(1)
            else:
                x_input = x_preprocessed

            if self.y_dimwise:
                assert self.x_dim == self.y_dim
                y_input = y_preprocessed[:, d].unsqueeze(1)
            else:
                y_input = y_preprocessed

            X = self.compute_basis(x_input, y_input)
            out.append(torch.from_numpy(self.models[k*self.x_dim+d].predict(X)))
        out = torch.cat(out, dim=1)
        return out.float()

    def fit(self, x, y, out, t):
        for k in range(self.num_steps):
            t_idx = torch.where(t[:, 0] == k)[0]

            self.x_means.append(x[t_idx].mean(0))
            self.x_stds.append(x[t_idx].std(0))
            self.y_means.append(y[t_idx].mean(0))
            self.y_stds.append(y[t_idx].std(0))

            x_preprocessed = (x[t_idx] - self.x_means[k]) / self.x_stds[k]
            y_preprocessed = (y[t_idx] - self.y_means[k]) / self.y_stds[k]

            for d in range(self.x_dim):
                if self.x_dimwise:
                    x_input = x_preprocessed[:, d].unsqueeze(1)
                else:
                    x_input = x_preprocessed

                if self.y_dimwise:
                    assert self.x_dim == self.y_dim
                    y_input = y_preprocessed[:, d].unsqueeze(1)
                else:
                    y_input = y_preprocessed

                X = self.compute_basis(x_input, y_input)
                self.models[k*self.x_dim+d].fit(X, out[t_idx, d].unsqueeze(1))
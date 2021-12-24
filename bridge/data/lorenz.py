import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.distributions import Normal, Independent
from scipy.integrate import solve_ivp, odeint
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from bridge.models.cond import BootstrapParticleFilter
from utils import mean_rmse

def RK4_step(f, t, x, h):
    k1 = h * f(t, x)
    k2 = h * f(t + 0.5 * h, x + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, x + 0.5 * k2)
    k4 = h * f(t + h, x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def RK45_step(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + 1. / 5 * h, x + h * (1. / 5 * k1))
    k3 = f(t + 3. / 10 * h, x + h * (3. / 40 * k1 + 9. / 40 * k2))
    k4 = f(t + 4. / 5 * h, x + h * (44. / 45 * k1 - 56. / 15 * k2 + 32. / 9 * k3))
    k5 = f(t + 8. / 9 * h, x + h * (19372. / 6561 * k1 - 25360. / 2187 * k2 + 64448. / 6561 * k3 - 212. / 729 * k4))
    k6 = f(t + h,
           x + h * (9017. / 3168 * k1 - 355. / 33 * k2 + 46732. / 5247 * k3 + 49. / 176 * k4 - 5103. / 18656 * k5))

    v5 = 35. / 384 * k1 + 500. / 1113 * k3 + 125. / 192 * k4 - 2187. / 6784 * k5 + 11. / 84 * k6
    # k7 = f(t + h, x + h * v5)
    # v4 = 5179. / 57600 * k1 + 7571. / 16695 * k3 + 393. / 640 * k4 - 92097. / 339200 * k5 + 187. / 2100 * k6 + 1. / 40 * k7
    # return v4, v5
    return x + h*v5


def forward_sim(x_0, data):
    F_fn, G_fn = forward_dist_fn(data)
    x = F_fn(x_0, None).sample()
    y = G_fn(x, None).sample()
    return x, y


def forward_dist_fn(data):
    if data == 'type1':
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        dt = 0.1
        delta = 0.05
        y_std = 2

        def f(t, state):
            f0 = sigma * (state[..., 1] - state[..., 0])
            f1 = state[..., 0] * (rho - state[..., 2]) - state[..., 1]
            f2 = state[..., 0] * state[..., 1] - beta * state[..., 2]
            return torch.stack([f0, f1, f2], -1)

        def F_fn(x, t):
            for _ in range(int(dt/delta)):
                x = RK45_step(f, 0, x, delta)
            return Independent(Normal(x, 0.01), 1)

        G_fn = lambda x, t: Independent(Normal(x, y_std * torch.ones_like(x)), 1)

    return F_fn, G_fn


def data_distrib(data):
    if data == 'type1':
        T = 4000
        x_t = torch.tensor([3., -3., 12.]).view(1, 3)
        x = torch.zeros([0, 3])
        y = torch.zeros([0, 3])
        for n in range(T):
            x_t, y_t = forward_sim(x_t, data)
            x = torch.cat([x, x_t], 0)
            y = torch.cat([y, y_t], 0)
    return x, y


def lorenz_process(root, data_tag):
    data_path = os.path.join(root, data_tag, "data.pt")
    if os.path.isfile(data_path):
        x, y = torch.load(data_path)
        print("Loaded dataset lorenz", data_tag)
    else:
        os.makedirs(os.path.join(root, data_tag), exist_ok=True)
        x, y = data_distrib(data_tag)
        torch.save([x, y], data_path)
        print("Created new dataset lorenz", data_tag)

    gt_filter_path = os.path.join(root, data_tag, "gt_filter.pt")
    if os.path.isfile(gt_filter_path):
        gt_means, gt_stds = torch.load(gt_filter_path)
    else:
        T, xdim, ydim = x.shape[0], x.shape[1], y.shape[1]
        x_0_mean = torch.tensor([3., -3., 12.])
        x_0_std = torch.ones([xdim])

        # BPF
        gt_means = torch.zeros([0, xdim])
        gt_stds = torch.zeros([0, xdim])

        p_0_dist = lambda: Independent(Normal(x_0_mean, x_0_std), 1)
        F_fn, G_fn = forward_dist_fn(data_tag)
        BPF = BootstrapParticleFilter(xdim, ydim, F_fn, G_fn, p_0_dist, 100000)

        for t in tqdm(range(T)):
            BPF.advance_timestep(y[t])
            BPF.update(y[t])

            gt_mean, gt_cov = BPF.return_summary_stats()
            gt_std = torch.diagonal(gt_cov).sqrt()

            gt_means = torch.vstack([gt_means, gt_mean])
            gt_stds = torch.vstack([gt_stds, gt_std])

        torch.save([gt_means, gt_stds], gt_filter_path)
    print("Mean RMSE (BPF):", mean_rmse(x, gt_means).numpy())

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(x[:, 0].numpy(), x[:, 1].numpy(), x[:, 2].numpy())
    # ax.plot(y[:, 0], y[:, 1], y[:, 2])
    plt.draw()
    plt.savefig(os.path.join(root, data_tag, "data.png"))
    plt.close()

    return x, y, gt_means, gt_stds


def lorenz_ds(x_0, data_tag):
    init_sample_x, init_sample_y = forward_sim(x_0, data_tag)
    init_ds = TensorDataset(init_sample_x, init_sample_y)
    return init_ds


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x, y = data_distrib('type1')
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.plot(y[:, 0], y[:, 1], y[:, 2])
    plt.draw()
    plt.show()

import numpy as np
import torch
import torch.nn.functional as functional
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


def forward_sim(x_0, data, args):
    F_fn, G_fn = forward_dist_fn(data, args)
    x = F_fn(x_0, None).sample()
    y = G_fn(x, None).sample()
    return x, y


def forward_dist_fn(data, args):
    if data == 'type1':
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        dt = 0.1
        delta = 0.05
        x_std = args.data.x_std
        y_std = args.data.y_std

        def f(t, x):
            f0 = sigma * (x[..., 1] - x[..., 0])
            f1 = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
            f2 = x[..., 0] * x[..., 1] - beta * x[..., 2]
            return torch.stack([f0, f1, f2], -1)

        def F_fn(x, t):
            for _ in range(int(dt/delta)):
                x = RK45_step(f, 0, x, delta)
            return Independent(Normal(x, x_std), 1)

        class G_module:
            def __init__(self):
                self.G = torch.eye(3)
                self.V = torch.eye(3) * y_std**2

            def __call__(self, x, t):
                return Independent(Normal(x, y_std), 1)

        G_fn = G_module()

    elif data == 'type2':
        xdim = args.x_dim
        ydim = args.y_dim
        forcing = 8
        dt = 0.4
        delta = 0.01
        if xdim % ydim == 0:
            obs_idx = np.arange(ydim) * (xdim // ydim)
        else:
            obs_idx = np.sort(np.random.choice(xdim, ydim, replace=False))
        G = torch.zeros((ydim, xdim))
        G[torch.arange(ydim), obs_idx] = 1
        x_std = args.data.x_std
        y_std = args.data.y_std

        def f(t, x):
            i = np.arange(xdim)
            d = (x[..., (i + 1) % xdim] - x[..., i - 2]) * x[..., i - 1] - x + forcing
            return d

        def F_fn(x, t):
            for _ in range(int(dt/delta)):
                x = RK4_step(f, 0, x, delta)
            return Independent(Normal(x, x_std, validate_args=False), 1)

        class G_module:
            def __init__(self):
                self.G = G
                self.V = torch.eye(ydim) * y_std**2

            def __call__(self, x, t):
                return Independent(Normal(functional.linear(x, G), y_std), 1)

        G_fn = G_module()

    return F_fn, G_fn


def data_distrib(data, args):
    T = args.data.T
    xdim = args.x_dim
    ydim = args.y_dim

    x = torch.zeros([0, xdim])
    y = torch.zeros([0, ydim])
    x_t = eval(args.data.x_0_mean).unsqueeze(0)

    for n in range(T):
        x_t, y_t = forward_sim(x_t, data, args)
        x = torch.cat([x, x_t], 0)
        y = torch.cat([y, y_t], 0)

    return x, y


def lorenz_process(root, data_tag, args):
    data_path = os.path.join(root, data_tag, "data.pt")
    if args.load and os.path.isfile(data_path):
        x, y = torch.load(data_path)
        print("Loaded dataset lorenz", data_tag)
        assert x.shape[0] == y.shape[0] == args.data.T
        assert x.shape[1] == args.x_dim
        assert y.shape[1] == args.y_dim
    else:
        os.makedirs(os.path.join(root, data_tag), exist_ok=True)
        x, y = data_distrib(data_tag, args)
        torch.save([x, y], data_path)
        print("Created new dataset lorenz", data_tag)

    gt_filter_path = os.path.join(root, data_tag, "gt_filter.pt")
    if args.load and os.path.isfile(gt_filter_path):
        gt_means, gt_stds = torch.load(gt_filter_path)
        assert gt_means.shape[0] == gt_stds.shape[0] == args.data.T
        assert gt_means.shape[1] == gt_stds.shape[1] == args.x_dim
    else:
        T, xdim, ydim = x.shape[0], x.shape[1], y.shape[1]
        x_0_mean = eval(args.data.x_0_mean)
        x_0_std = eval(args.data.x_0_std)

        if data_tag == 'type1':
            # BPF
            gt_means = torch.zeros([0, xdim])
            gt_stds = torch.zeros([0, xdim])

            p_0_dist = lambda: Independent(Normal(x_0_mean, x_0_std), 1)
            F_fn, G_fn = forward_dist_fn(data_tag, args)
            BPF = BootstrapParticleFilter(xdim, ydim, F_fn, G_fn, p_0_dist, 100000)

            for t in tqdm(range(T)):
                BPF.advance_timestep(y[t])
                BPF.update(y[t])

                gt_mean, gt_cov = BPF.return_summary_stats()
                gt_std = torch.diagonal(gt_cov).sqrt()

                gt_means = torch.vstack([gt_means, gt_mean])
                gt_stds = torch.vstack([gt_stds, gt_std])

            torch.save([gt_means, gt_stds], gt_filter_path)
        else:
            gt_means = torch.zeros_like(x)
            gt_stds = torch.zeros_like(x)

    print("Mean RMSE (BPF):", mean_rmse(x, gt_means).numpy())

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot(x[:, 0].numpy(), x[:, 1].numpy(), x[:, 2].numpy())
    # ax.plot(y[:, 0], y[:, 1], y[:, 2])
    plt.draw()
    plt.savefig(os.path.join(root, data_tag, "data.png"))
    plt.close()

    return x, y, gt_means, gt_stds


def lorenz_ds(x_0, data_tag, args):
    init_sample_x, init_sample_y = forward_sim(x_0, data_tag, args)
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

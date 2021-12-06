import torch
from torch.utils.data import TensorDataset
from scipy.integrate import solve_ivp, odeint
import os


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
    if data == 'type1':
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        dt = 0.01
        y_std = 2

        def f(t, state):
            f0 = sigma * (state[..., 1] - state[..., 0])
            f1 = state[..., 0] * (rho - state[..., 2]) - state[..., 1]
            f2 = state[..., 0] * state[..., 1] - beta * state[..., 2]
            return torch.stack([f0, f1, f2], -1)

        x = RK45_step(f, 0, x_0, dt)
        y = x + torch.randn(*x.shape) * y_std

    return x, y


def data_distrib(data):
    if data == 'type1':
        T = 2000
        x_t = torch.ones([1, 3])
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
    return x, y


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

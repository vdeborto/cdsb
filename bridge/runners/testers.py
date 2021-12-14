import numpy as np
import torch
import torch.nn.functional as functional
import os
from scipy.stats import kde, gamma, norm
from tqdm import tqdm


class Tester:
    def __init__(self):
        pass

    def test_joint(self, *args, **kwargs):
        return {}

    def test_cond(self, *args, **kwargs):
        return {}

    def __call__(self, x_start, y_start, x_tot, y_cond, x_tot_cond, x_init, data, i, n, fb, x_init_cond=None):
        x_last = x_tot[-1]

        x_var_last = torch.var(x_last).item()
        x_var_start = torch.var(x_start).item()
        x_mean_last = torch.mean(x_last).item()
        x_mean_start = torch.mean(x_start).item()

        out = {'FB': fb,
               'x_mean_start': x_mean_start, 'x_var_start': x_var_start,
               'x_mean_last': x_mean_last, 'x_var_last': x_var_last}

        out.update(self.test_joint(y_start, x_tot, x_init, data, i, n, fb))
        out.update(self.test_cond(y_cond, x_tot_cond, data, i, n, fb, x_init_cond=x_init_cond))

        return out


class OneDCondTester(Tester):
    def __init__(self):
        super().__init__()

    def test_joint(self, y_start, x_tot, x_init, data, i, n, fb, tag=''):
        out = {}

        if fb == 'b':
            y_start = y_start.detach().cpu().numpy()
            x_last = x_tot[-1].detach().cpu().numpy()
            last_kde = lambda xy: kde.gaussian_kde([x_last[:, 0], y_start[:, 0]])(xy.T)

            x_init = x_init.detach().cpu().numpy()
            data_kde = lambda xy: kde.gaussian_kde([x_init[:, 0], y_start[:, 0]])(xy.T)

            batch = np.hstack([x_init, y_start])

            out["l2_pq_" + tag] = np.mean((data_kde(batch) - last_kde(batch)) ** 2)
            out["kl_pq_" + tag] = np.mean(np.log(data_kde(batch)) - np.log(last_kde(batch)))

        return out


class FiveDCondTester(Tester):
    def __init__(self):
        super().__init__()

    def test_cond(self, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag=''):
        out = {}

        if fb == 'b' and y_cond is not None:
            if data == 'type1':
                true_x_test_mean = (y_cond[:, 0]**2 + torch.exp(y_cond[:, 1] + y_cond[:, 2]/3) + torch.sin(y_cond[:, 3] + y_cond[:, 4])).unsqueeze(1)
                true_x_test_std = torch.ones(2000, 1)
            
            elif data == 'type2':
                true_x_test_mean = (y_cond[:, 0]**2 + torch.exp(y_cond[:, 1] + y_cond[:, 2]/3) + y_cond[:, 3] - y_cond[:, 4]).unsqueeze(1)
                true_x_test_std = (0.5 + y_cond[:, 1]**2/2 + y_cond[:, 4]**2/2).unsqueeze(1)

            elif data == 'type3':
                mult = (5 + y_cond[:, 0]**2/3 + y_cond[:, 1]**2 + y_cond[:, 2]**2 + y_cond[:, 3] + y_cond[:, 4]).unsqueeze(1)
                log_normal_mix_mean = 0.5 * np.exp(1 + 0.5**2/2) + 0.5 * np.exp(-1 + 0.5**2/2)
                true_x_test_mean = mult * log_normal_mix_mean
                true_x_test_std = mult * np.sqrt(0.5 * np.exp(2 + 2*0.5**2) + 0.5 * np.exp(-2 + 2*0.5**2) - log_normal_mix_mean**2)

            elif data == 'type4':
                true_x_test_mean = torch.zeros(2000, 1)
                true_x_test_std = np.sqrt(y_cond[:, 0:1]**2 + 0.25**2)

            x_tot_cond_std, x_tot_cond_mean = torch.std_mean(x_tot_cond[-1], 1)

            out["mse_mean_" + tag] = torch.mean((x_tot_cond_mean - true_x_test_mean)**2)
            out["mse_std_" + tag] = torch.mean((x_tot_cond_std - true_x_test_std)**2)

        return out


if __name__ == "__main__":
    # x_train = np.random.randn(50, 1)
    # y_train = np.random.randn(50, 1)
    # x_test = np.random.randn(50, 1)
    # y_cond = np.random.randn(50, 1)
    # xy_cond = np.hstack([x_test, y_cond])
    # kde1 = kde.gaussian_kde([x_train[:, 0], y_train[:, 0]])
    # kde2 = kde.gaussian_kde(np.hstack([x_test, y_cond]).T)
    # # assert np.array_equal(kde1, kde2)
    # # print("Done")
    # l2 = kde1.integrate_kde(kde1) + kde2.integrate_kde(kde2) - 2 * kde1.integrate_kde(kde2)
    # print(l2)
    npar = 1000000
    y = torch.randn(5).expand(npar, 5)
    data = 'type4'
    
    if data == 'type1':
        normal = torch.randn(npar, 1)
        x_mean = y[:, 0]**2 + torch.exp(y[:, 1] + y[:, 2]/3) + torch.sin(y[:, 3] + y[:, 4])
        x = x_mean.reshape(npar, 1) + normal

    elif data == 'type2':
        normal = torch.randn(npar, 1)
        x_mean = y[:, 0]**2 + torch.exp(y[:, 1] + y[:, 2]/3) + y[:, 3] - y[:, 4]
        x_std = 0.5 + y[:, 1]**2/2 + y[:, 4]**2/2
        x = x_mean.reshape(npar, 1) + normal * x_std.reshape(npar, 1)

    elif data == 'type3':
        uniform = torch.rand(npar, 1)
        normal_n = torch.randn(npar, 1) - 2
        normal_p = torch.randn(npar, 1) + 2
        eps = torch.where(uniform < 0.5, normal_n, normal_p)
        x = (5 + y[:, 0]**2/3 + y[:, 1]**2 + y[:, 2]**2 + y[:, 3] + y[:, 4]).reshape(npar, 1) * torch.exp(0.5*eps)

    elif data == 'type4':
        uniform = torch.rand(npar, 1)
        normal_n = torch.randn(npar, 1)*0.25 - y[:, 0:1]
        normal_p = torch.randn(npar, 1)*0.25 + y[:, 0:1]
        x = torch.where(uniform < 0.5, normal_n, normal_p)
    
    y_cond = y
    if data == 'type1':
        true_x_test_mean = (y_cond[:, 0]**2 + torch.exp(y_cond[:, 1] + y_cond[:, 2]/3) + torch.sin(y_cond[:, 3] + y_cond[:, 4])).unsqueeze(1)
        true_x_test_std = torch.ones(2000, 1)
    
    elif data == 'type2':
        true_x_test_mean = (y_cond[:, 0]**2 + torch.exp(y_cond[:, 1] + y_cond[:, 2]/3) + y_cond[:, 3] - y_cond[:, 4]).unsqueeze(1)
        true_x_test_std = (0.5 + y_cond[:, 1]**2/2 + y_cond[:, 4]**2/2).unsqueeze(1)

    elif data == 'type3':
        mult = (5 + y_cond[:, 0]**2/3 + y_cond[:, 1]**2 + y_cond[:, 2]**2 + y_cond[:, 3] + y_cond[:, 4]).unsqueeze(1)
        log_normal_mix_mean = 0.5 * np.exp(1 + 0.5**2/2) + 0.5 * np.exp(-1 + 0.5**2/2)
        true_x_test_mean = mult * log_normal_mix_mean
        true_x_test_std = mult * np.sqrt(0.5 * (np.exp(1 + 0.5**2/2)**2 + (np.exp(0.5**2)-1)*np.exp(2+0.5**2)) + 
                                            0.5 * (np.exp(-1 + 0.5**2/2)**2 + (np.exp(0.5**2)-1)*np.exp(-2+0.5**2)) - 
                                            log_normal_mix_mean**2)

    elif data == 'type4':
        true_x_test_mean = torch.zeros(2000, 1)
        true_x_test_std = np.sqrt(y_cond[:, 0:1]**2 + 0.25**2)
    
    print(x.mean(dim=0), true_x_test_mean[0])
    print(x.std(dim=0), true_x_test_std[0])

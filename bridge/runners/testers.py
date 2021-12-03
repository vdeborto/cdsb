import numpy as np
import torch
import torch.nn.functional as functional
import os
from scipy.stats import kde, gamma, norm
from tqdm import tqdm


class Tester:
    def __init__(self):
        pass


class OneDCondTester(Tester):
    def __init__(self):
        super().__init__()

    def test_joint(self, x_tot, y_tot, data, init_dl, i, n, fb, tag=''):
        out = {}

        if fb == 'b':
            x_final = x_tot[-1]
            y_final = y_tot[-1]

            x_final = x_final.detach().cpu().numpy()
            y_final = y_final.detach().cpu().numpy()

            final_kde = lambda xy: kde.gaussian_kde([x_final[:, 0], y_final[:, 0]])(xy.T)

            batch = next(init_dl)
            x_batch = batch[0].cpu().numpy()
            y_batch = batch[1].cpu().numpy()
            true_kde = lambda xy: kde.gaussian_kde([x_batch[:, 0], y_batch[:, 0]])(xy.T)

            batch = np.hstack([x_batch, y_batch])

            out["l2_pq_" + tag] = np.mean((true_kde(batch) - final_kde(batch))**2)
            out["kl_pq_" + tag] = np.mean(np.log(true_kde(batch)) - np.log(final_kde(batch)))

        return out

    def test_cond(self, y_cond, x_tot_cond, data, i, n, fb, tag=''):
        out = {}
        return out

    def __call__(self, x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, init_dl, i, n, fb):
        x_final = x_tot[-1]
        y_final = y_tot[-1]
        
        x_var_final = torch.var(x_final)
        x_var_init = torch.var(x_init)
        x_mean_final = torch.mean(x_final)
        x_mean_init = torch.mean(x_init)

        out = {'FB': fb,
               'x_mean_init': x_mean_init, 'x_var_init': x_var_init,
               'x_mean_final': x_mean_final, 'x_var_final': x_var_final}

        out.update(self.test_joint(x_tot, y_tot, data, init_dl, i, n, fb))
        out.update(self.test_cond(y_cond, x_tot_cond, data, i, n, fb))

        return out


class FiveDCondTester(Tester):
    def __init__(self):
        super().__init__()

    def test_joint(self, x_tot, y_tot, data, init_dl, i, n, fb, tag=''):
        out = {}
        return out

    def test_cond(self, y_cond, x_tot_cond, data, i, n, fb, tag=''):
        out = {}

        if fb == 'b':
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

    def __call__(self, x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, init_dl, i, n, fb):
        x_final = x_tot[-1]
        y_final = y_tot[-1]
        
        x_var_final = torch.var(x_final)
        x_var_init = torch.var(x_init)
        x_mean_final = torch.mean(x_final)
        x_mean_init = torch.mean(x_init)

        out = {'FB': fb,
               'x_mean_init': x_mean_init, 'x_var_init': x_var_init,
               'x_mean_final': x_mean_final, 'x_var_final': x_var_final}

        out.update(self.test_joint(x_tot, y_tot, data, init_dl, i, n, fb))
        out.update(self.test_cond(y_cond, x_tot_cond, data, i, n, fb))
        
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

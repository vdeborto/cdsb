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

    def test(self, x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, save_init_dl, i, n, fb):
        x_tot = x_tot.cpu().numpy()
        y_tot = y_tot.cpu().numpy()

        x_init = x_init.detach().cpu().numpy()
        x_final = x_tot[-1]
        y_final = y_tot[-1]
        
        x_var_final = np.var(x_final)
        x_var_init = np.var(x_init)
        x_mean_final = np.mean(x_final)
        x_mean_init = np.mean(x_init)

        out = {'FB': fb,
               'x_mean_init': x_mean_init, 'x_var_init': x_var_init,
               'x_mean_final': x_mean_final, 'x_var_final': x_var_final}

        if fb == 'b':
            final_kde = lambda xy: kde.gaussian_kde([x_final[:, 0], y_final[:, 0]])(xy.T)

            batch = next(save_init_dl)
            x_batch = batch[0].cpu().numpy()
            y_batch = batch[1].cpu().numpy()
            true_kde = lambda xy: kde.gaussian_kde([x_batch[:, 0], y_batch[:, 0]])(xy.T)

            batch = np.hstack([x_batch, y_batch])

            out["l2_pq"] = np.mean((true_kde(batch) - final_kde(batch))**2)
            out["kl_pq"] = np.mean(np.log(true_kde(batch)) - np.log(final_kde(batch)))

        return out

    def __call__(self, x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, save_init_dl, i, n, fb):
        return self.test(x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, save_init_dl, i, n, fb)


class FiveDCondTester(Tester):
    def __init__(self, runner):
        super().__init__()
        self.runner = runner

    def test(self, x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, save_init_dl, i, n, fb):
        x_tot_np = x_tot.cpu().numpy()

        x_init_np = x_init.detach().cpu().numpy()
        x_final_np = x_tot_np[-1]
        
        x_var_final = np.var(x_final_np)
        x_var_init = np.var(x_init_np)
        x_mean_final = np.mean(x_final_np)
        x_mean_init = np.mean(x_init_np)

        out = {'FB': fb,
               'x_mean_init': x_mean_init, 'x_var_init': x_var_init,
               'x_mean_final': x_mean_final, 'x_var_final': x_var_final}

        if fb == 'b':
            y_test = torch.randn(2000, 5)
            if data == 'type1':
                true_x_test_mean = (y_test[:, 0]**2 + torch.exp(y_test[:, 1] + y_test[:, 2]/3) + torch.sin(y_test[:, 3] + y_test[:, 4])).unsqueeze(1)
                true_x_test_std = torch.ones(2000, 1)
            
            elif data == 'type2':
                true_x_test_mean = (y_test[:, 0]**2 + torch.exp(y_test[:, 1] + y_test[:, 2]/3) + y_test[:, 3] - y_test[:, 4]).unsqueeze(1)
                true_x_test_std = (0.5 + y_test[:, 1]**2/2 + y_test[:, 4]**2/2).unsqueeze(1)

            elif data == 'type3':
                mult = (5 + y_test[:, 0]**2/3 + y_test[:, 1]**2 + y_test[:, 2]**2 + y_test[:, 3] + y_test[:, 4]).unsqueeze(1)
                log_normal_mix_mean = 0.5 * np.exp(1 + 0.5**2/2) + 0.5 * np.exp(-1 + 0.5**2/2)
                true_x_test_mean = mult * log_normal_mix_mean
                true_x_test_std = mult * np.sqrt(0.5 * np.exp(2 + 2*0.5**2) + 0.5 * np.exp(-2 + 2*0.5**2) - log_normal_mix_mean**2)

            elif data == 'type4':
                true_x_test_mean = torch.zeros(2000, 1)
                true_x_test_std = np.sqrt(y_test[:, 0:1]**2 + 0.25**2)


            if self.runner.args.ema:
                sample_net = self.runner.ema_helpers[fb].ema_copy(self.runner.net[fb])
            else:
                sample_net = self.runner.net[fb]

            shape_len = len(x_tot.shape)
            x_tot_cond = torch.zeros([0, *x_tot.shape])

            for k in tqdm(range(len(y_test))):
                y_c = y_test[k].expand_as(y_init).to(self.runner.device)
                x_tot_c, _, _, _ = self.runner.langevin.record_langevin_seq(sample_net, x_init, y_c, sample=True)

                x_tot_c = x_tot_c.permute(1, 0, *list(range(2, shape_len)))
                x_tot_c_plot = x_tot_c.detach()#.cpu().numpy()
                x_tot_cond = torch.cat([x_tot_cond, x_tot_c_plot.unsqueeze(0)], dim=0)
            x_tot_cond_std, x_tot_cond_mean = torch.std_mean(x_tot_cond[:, -1], 1)

            out["mse_mean"] = torch.mean((x_tot_cond_mean - true_x_test_mean)**2)
            out["mse_std"] = torch.mean((x_tot_cond_std - true_x_test_std)**2)

        return out

    def __call__(self, x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, save_init_dl, i, n, fb):
        return self.test(x_init, y_init, x_tot, y_tot, x_tot_cond, y_cond, data, save_init_dl, i, n, fb)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as functional
import os
from scipy.stats import kde, gamma, norm

matplotlib.use('Agg')


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
        
        x_final_std = np.std(x_final)
        x_init_std = np.std(x_init)
        x_final_mean = np.mean(x_final)
        x_init_mean = np.mean(x_init)

        # print('Initial variance: ' + str(std_init ** 2))
        # print('Final variance: ' + str(std_final ** 2))

        out = {'FB': fb,
               'x_init_mean': x_init_mean, 'x_init_std': x_init_std,
               'x_final_mean': x_final_mean, 'x_final_std': x_final_std}

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

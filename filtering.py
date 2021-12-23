import numpy as np
import torch
from torch.distributions import Normal, Independent
import hydra
import os, sys
from matplotlib import pyplot as plt

sys.path.append('..')

from bridge.runners.ipf import IPFSequential
from bridge.runners.config_getters import get_filtering_datasets, get_filtering_process
from bridge.data.lorenz import forward_dist_fn
from bridge.models.cond import EnsembleKalmanFilter


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="filtering")
def main(args):
    # torch.set_default_dtype(torch.float64)

    print('Directory: ' + os.getcwd())
    os.mkdir("im")

    x, y = get_filtering_process(args)
    x_np = x.detach().cpu().numpy()
    T, xdim = x.shape
    ydim = y.shape[1]

    x_0_mean = torch.ones([xdim])
    x_0_std = torch.ones([xdim])
    x_ens = torch.randn([args.ens_size, xdim]) * x_0_std + x_0_mean

    x_ens_means_enkf = np.zeros([0, xdim])
    x_ens_stds_enkf = np.zeros([0, xdim])
    rmses_enkf = np.zeros([0])

    # EnKF
    p_0_dist = lambda: Independent(Normal(x_0_mean, x_0_std), 1)
    F_fn, G_fn = forward_dist_fn(args.data.dataset)
    EnKF = EnsembleKalmanFilter(xdim, ydim, F_fn, G_fn, p_0_dist, args.ens_size)

    for t in range(T):
        EnKF.advance_timestep(y[t])
        EnKF.update(y[t])

        x_ens_mean, x_ens_cov = EnKF.return_summary_stats()
        x_ens_std = torch.diagonal(x_ens_cov).sqrt()

        x_ens_means_enkf = np.row_stack([x_ens_means_enkf, x_ens_mean.numpy()])
        x_ens_stds_enkf = np.row_stack([x_ens_stds_enkf, x_ens_std.numpy()])
        rmses_enkf = np.append(rmses_enkf, np.sqrt(np.mean((x_ens_means_enkf[-1] - x_np[t]) ** 2)))

    np.save("x_ens_means_enkf.npy", x_ens_means_enkf)
    np.save("x_ens_stds_enkf.npy", x_ens_stds_enkf)
    np.save("rmses_enkf.npy", rmses_enkf)

    mean_rmse_enkf = np.mean(rmses_enkf[T // 10:])
    print("Mean RMSE (EnKF):", mean_rmse_enkf)

    plt.clf()
    fig = plt.figure(figsize=(15, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(np.arange(T), x[:, i], color="C0")
        plt.plot(np.arange(T), y[:, i], 'o', color="C1")
        plt.plot(np.arange(T), x_ens_means_enkf[:, i], '--', color="C2")
        plt.fill_between(np.arange(T), x_ens_means_enkf[:, i] - x_ens_stds_enkf[:, i], x_ens_means_enkf[:, i] + x_ens_stds_enkf[:, i],
                         alpha=0.2, color="C2")
    plt.savefig("im/filter_mean_std_EnKF.png")

    plt.clf()
    plt.plot(np.arange(T), rmses_enkf)
    plt.savefig("im/filter_rmse_EnKF.png")


    # IPF
    x_ens_means = np.zeros([0, xdim])
    x_ens_stds = np.zeros([0, xdim])
    rmses = np.zeros([0])

    EnKF = EnsembleKalmanFilter(xdim, ydim, F_fn, G_fn, p_0_dist, args.ens_size)

    for t in range(T):
        EnKF.advance_timestep(y[t])
        EnKF.update(y[t])

        with torch.no_grad():
            x_ens_repeat = x_ens.repeat(args.npar//args.ens_size, 1)
            init_ds_repeat, final_ds_repeat, mean_final, var_final = get_filtering_datasets(x_ens_repeat, args)

        ipf = IPFSequential(init_ds_repeat, final_ds_repeat, mean_final, var_final, args, final_cond_model=EnKF)
        if t == 0:
            ipf.accelerator.print(ipf.accelerator.state)
            ipf.accelerator.print(ipf.net['b'])
            ipf.accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
        ipf.train()

        if ipf.accelerator.is_main_process:
            with torch.no_grad():
                if args.cond_final:
                    mean_final, std_final = EnKF(y[t].to(ipf.device))
                    final_x = mean_final + std_final * torch.randn(x_ens.shape).to(ipf.device)
                    x_ens = ipf.backward_sample(final_x, y[t])[-1].cpu()

                    EnKF.x_T = x_ens
                else:
                    init_ds, _, _, _ = get_filtering_datasets(x_ens, args)
                    init_x, init_y = init_ds.tensors
                    x_ens = ipf.forward_backward_sample(init_x, init_y, y[t], n=args.n_ipf)[-1].cpu()

            x_ens_means = np.row_stack([x_ens_means, x_ens.mean(0).numpy()])
            x_ens_stds = np.row_stack([x_ens_stds, x_ens.std(0).numpy()])
            rmses = np.append(rmses, np.sqrt(np.mean((x_ens_means[-1] - x_np[t]) ** 2)))

            ipf.accelerator.print(x[t].numpy())
            ipf.accelerator.print(x_ens_means[-1], x_ens_stds[-1])
            ipf.accelerator.print("RMSE:", rmses[-1])

            plt.clf()
            fig = plt.figure(figsize=(15, 6))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.plot(np.arange(t+1), x[:t+1, i], color="C0")
                plt.plot(np.arange(t+1), y[:t+1, i], 'o', color="C1")
                plt.plot(np.arange(t+1), x_ens_means[:, i], '--', color="C2")
                plt.fill_between(np.arange(t+1), x_ens_means[:, i] - x_ens_stds[:, i], x_ens_means[:, i] + x_ens_stds[:, i],
                                 alpha=0.2, color="C2")
            plt.savefig("im/filter_mean_std.png")

            plt.clf()
            plt.plot(np.arange(t+1), rmses)
            plt.savefig("im/filter_rmse.png")

            np.save("x_ens_means.npy", x_ens_means)
            np.save("x_ens_stds.npy", x_ens_stds)
            np.save("rmses.npy", rmses)

        ipf.accelerator.free_memory()
        del ipf

    mean_rmse = np.mean(rmses[T // 10:])
    print("Mean RMSE:", mean_rmse)

if __name__ == '__main__':
    main()

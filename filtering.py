import numpy as np
import torch
from torch.distributions import Normal, Independent
import hydra
import os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append('..')

from bridge.runners.ipf import IPFSequential, IPFAnalytic
from bridge.runners.config_getters import get_filtering_datasets, get_filtering_process
from bridge.data.lorenz import forward_dist_fn
from bridge.models.cond import EnsembleKalmanFilter, EnsembleKalmanFilterSpinup


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="filtering")
def main(args):
    # torch.set_default_dtype(torch.float64)

    print('Directory: ' + os.getcwd())
    os.mkdir("im")

    x, y, gt_means, gt_stds = get_filtering_process(args)
    x_np = x.detach().cpu().numpy()
    gt_means_np, gt_stds_np = gt_means.detach().cpu().numpy(), gt_stds.detach().cpu().numpy()

    default_dtype = torch.get_default_dtype()
    x, y, gt_means, gt_stds = x.to(default_dtype), y.to(default_dtype), gt_means.to(default_dtype), gt_stds.to(default_dtype)
    T, xdim, ydim = x.shape[0], x.shape[1], y.shape[1]
    assert xdim == args.x_dim and ydim == args.y_dim

    T_spinup = T // 2    

    x_0_mean = eval(args.data.x_0_mean)
    x_0_std = eval(args.data.x_0_std)

    p_0_dist = lambda: Independent(Normal(x_0_mean, x_0_std), 1)
    F_fn, G_fn = forward_dist_fn(args.data.dataset, args)

    if args.EnKF_run:
        # EnKF
        x_ens_means_enkf = np.zeros([0, xdim])
        x_ens_stds_enkf = np.zeros([0, xdim])
        rmses_enkf = np.zeros([0])
        filter_rmses_enkf = np.zeros([0])
        filter_std_rmses_enkf = np.zeros([0])

        EnKF = EnsembleKalmanFilter(xdim, ydim, F_fn, G_fn, p_0_dist, args.ens_size)
        EnKF_spinup = EnsembleKalmanFilterSpinup(xdim, ydim, F_fn, G_fn, p_0_dist, args.ens_size)

        for t in tqdm(range(T)):
            if t < T_spinup:
                EnKF_spinup.advance_timestep(y[t])
                EnKF_spinup.update(y[t])

                x_ens_mean, x_ens_cov = EnKF_spinup.return_summary_stats()
                x_ens_std = torch.diagonal(x_ens_cov).sqrt()

                if t == T_spinup - 1:
                    print("Mean RMSE (spinup):", np.mean(rmses_enkf))
                    print("Mean filter RMSE (spinup):", np.mean(filter_rmses_enkf))

                    EnKF.x_T = EnKF_spinup.x_T
                    EnKF.T = t
            else:
                EnKF.advance_timestep(y[t])
                EnKF.update(y[t])

                x_ens_mean, x_ens_cov = EnKF.return_summary_stats()
                x_ens_std = torch.diagonal(x_ens_cov).sqrt()

            x_ens_means_enkf = np.row_stack([x_ens_means_enkf, x_ens_mean.numpy()])
            x_ens_stds_enkf = np.row_stack([x_ens_stds_enkf, x_ens_std.numpy()])
            rmses_enkf = np.append(rmses_enkf, np.sqrt(np.mean((x_ens_means_enkf[-1] - x_np[t]) ** 2)))
            filter_rmses_enkf = np.append(filter_rmses_enkf, np.sqrt(np.mean((x_ens_means_enkf[-1] - gt_means_np[t]) ** 2)))
            filter_std_rmses_enkf = np.append(filter_std_rmses_enkf, np.sqrt(np.mean((x_ens_stds_enkf[-1] - gt_stds_np[t]) ** 2)))

        np.save("x_ens_means_enkf.npy", x_ens_means_enkf)
        np.save("x_ens_stds_enkf.npy", x_ens_stds_enkf)
        np.save("rmses_enkf.npy", rmses_enkf)
        np.save("filter_rmses_enkf.npy", filter_rmses_enkf)
        np.save("filter_std_rmses_enkf.npy", filter_std_rmses_enkf)

        mean_rmse_enkf = np.mean(rmses_enkf[T_spinup:])
        mean_filter_rmse_enkf = np.mean(filter_rmses_enkf[T_spinup:])
        print("Mean RMSE (EnKF):", mean_rmse_enkf)
        print("Mean filter RMSE (EnKF):", mean_filter_rmse_enkf)

        plt.clf()
        fig = plt.figure(figsize=(15, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.plot(np.arange(T_spinup, T), x[T_spinup:, i], color="C0")
            if xdim == ydim:
                plt.plot(np.arange(T_spinup, T), y[T_spinup:, i], 'o', color="C1")
            plt.plot(np.arange(T_spinup, T), x_ens_means_enkf[T_spinup:, i], '--', color="C2")
            plt.fill_between(np.arange(T_spinup, T),
                             x_ens_means_enkf[T_spinup:, i] - x_ens_stds_enkf[T_spinup:, i],
                             x_ens_means_enkf[T_spinup:, i] + x_ens_stds_enkf[T_spinup:, i],
                             alpha=0.2, color="C2")
            plt.plot(np.arange(T_spinup, T), gt_means_np[T_spinup:, i], ':', color="C3")
            plt.fill_between(np.arange(T_spinup, T),
                             gt_means_np[T_spinup:, i] - gt_stds_np[T_spinup:, i],
                             gt_means_np[T_spinup:, i] + gt_stds_np[T_spinup:, i],
                             alpha=0.2, color="C3")
        plt.savefig("im/filter_mean_std_enkf.png")

        plt.clf()
        plt.plot(np.arange(T_spinup, T), rmses_enkf[T_spinup:], label=f'EnKF (RMSE {np.mean(rmses_enkf[T_spinup:])})')
        plt.legend()
        plt.savefig("im/rmse_enkf.png")

        plt.clf()
        plt.plot(np.arange(T_spinup, T), filter_rmses_enkf[T_spinup:], label=f'EnKF (Filter RMSE {np.mean(filter_rmses_enkf[T_spinup:])})')
        plt.legend()
        plt.savefig("im/filter_rmses_enkf.png")

        plt.clf()
        plt.plot(np.arange(T_spinup, T), filter_std_rmses_enkf[T_spinup:], label=f'EnKF (Filter std RMSE {np.mean(filter_std_rmses_enkf[T_spinup:])})')
        plt.legend()
        plt.savefig("im/filter_std_rmses_enkf.png")
        plt.close()


    else:
        # IPF
        x_ens_means = np.zeros([0, xdim])
        x_ens_stds = np.zeros([0, xdim])
        rmses = np.zeros([0])
        filter_rmses = np.zeros([0])
        filter_std_rmses = np.zeros([0])

        rmses_enkf = np.zeros([0])
        filter_rmses_enkf = np.zeros([0])
        filter_std_rmses_enkf = np.zeros([0])

        EnKF = EnsembleKalmanFilter(xdim, ydim, F_fn, G_fn, p_0_dist, args.ens_size, std_scale=args.cond_final_model.std_scale)
        EnKF_spinup = EnsembleKalmanFilterSpinup(xdim, ydim, F_fn, G_fn, p_0_dist, args.ens_size)

        for t in tqdm(range(T)):
            if t < T_spinup:
                EnKF_spinup.advance_timestep(y[t])
                EnKF_spinup.update(y[t])

                x_ens_mean, x_ens_cov = EnKF_spinup.return_summary_stats()
                x_ens_std = torch.diagonal(x_ens_cov).sqrt()

                x_ens_mean, x_ens_std = x_ens_mean.numpy(), x_ens_std.numpy()
                x_ens_means = np.row_stack([x_ens_means, x_ens_mean])
                x_ens_stds = np.row_stack([x_ens_stds, x_ens_std])
                rmses_enkf = np.append(rmses_enkf, np.sqrt(np.mean((x_ens_mean - x_np[t]) ** 2)))
                filter_rmses_enkf = np.append(filter_rmses_enkf, np.sqrt(np.mean((x_ens_mean - gt_means_np[t]) ** 2)))
                filter_std_rmses_enkf = np.append(filter_std_rmses_enkf, np.sqrt(np.mean((x_ens_std - gt_stds_np[t]) ** 2)))

                rmses = np.append(rmses, rmses_enkf[-1])
                filter_rmses = np.append(filter_rmses, filter_rmses_enkf[-1])
                filter_std_rmses = np.append(filter_std_rmses, filter_std_rmses_enkf[-1])

                if t == T_spinup - 1:
                    print("Mean RMSE (spinup):", np.mean(rmses))
                    print("Mean filter RMSE (spinup):", np.mean(filter_rmses))

                    EnKF.x_T = EnKF_spinup.x_T
                    EnKF.T = t
                    x_ens = EnKF.x_T

            else:
                EnKF.advance_timestep(y[t])
                EnKF.update(y[t])

                x_ens_mean, x_ens_cov = EnKF.return_summary_stats()
                x_ens_std = torch.diagonal(x_ens_cov).sqrt()

                x_ens_mean, x_ens_std = x_ens_mean.numpy(), x_ens_std.numpy()
                rmses_enkf = np.append(rmses_enkf, np.sqrt(np.mean((x_ens_mean - x_np[t]) ** 2)))
                filter_rmses_enkf = np.append(filter_rmses_enkf, np.sqrt(np.mean((x_ens_mean - gt_means_np[t]) ** 2)))
                filter_std_rmses_enkf = np.append(filter_std_rmses_enkf, np.sqrt(np.mean((x_ens_std - gt_stds_np[t]) ** 2)))

                with torch.no_grad():
                    x_ens_repeat = x_ens.repeat(args.npar//args.ens_size, 1)
                    init_ds_repeat, final_ds_repeat, mean_final, var_final = get_filtering_datasets(x_ens_repeat, args)
                print("True state:", x_np[t])
                print("Filter mean:", gt_means_np[t])
                print("Filter std:", gt_stds_np[t])
                print("Prior mean:", init_ds_repeat.tensors[0].mean(0).numpy())
                print("Prior std:", init_ds_repeat.tensors[0].std(0).numpy())

                if args.Model in ['PolyCond', 'BasisCond', 'KRRCond']:
                    ipf = IPFAnalytic(init_ds_repeat, final_ds_repeat, mean_final, var_final, args, final_cond_model=EnKF)
                else:
                    ipf = IPFSequential(init_ds_repeat, final_ds_repeat, mean_final, var_final, args, final_cond_model=EnKF)
                if t == T_spinup:
                    ipf.accelerator.print(ipf.net['b'])
                    ipf.accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
                ipf.train()

                if ipf.accelerator.is_main_process:
                    with torch.no_grad():
                        if args.cond_final:
                            mean_final, std_final = EnKF(y[t].to(ipf.device))
                            final_x = mean_final + std_final * torch.randn(x_ens.shape).to(ipf.device)
                            var_final = std_final ** 2
                            x_ens = ipf.backward_sample(final_x, y[t], var_final=var_final)[-1].cpu()
                        else:
                            init_ds, _, _, _ = get_filtering_datasets(x_ens, args)
                            init_x, init_y = init_ds.tensors
                            if args.fwd_bwd_sample:
                                x_ens = ipf.forward_backward_sample(init_x, init_y, y[t], args.n_ipf, 'f')[-1].cpu()
                            else:
                                std_final = torch.sqrt(var_final)
                                final_x = mean_final + std_final * torch.randn(x_ens.shape).to(ipf.device)
                                x_ens = ipf.backward_sample(final_x, y[t])[-1].cpu()

                    x_ens_means = np.row_stack([x_ens_means, x_ens.mean(0).numpy()])
                    x_ens_stds = np.row_stack([x_ens_stds, x_ens.std(0).numpy()])
                    rmses = np.append(rmses, np.sqrt(np.mean((x_ens_means[-1] - x_np[t]) ** 2)))
                    filter_rmses = np.append(filter_rmses, np.sqrt(np.mean((x_ens_means[-1] - gt_means_np[t]) ** 2)))
                    filter_std_rmses = np.append(filter_std_rmses, np.sqrt(np.mean((x_ens_stds[-1] - gt_stds_np[t]) ** 2)))

                    ipf.accelerator.print(x[t].numpy())
                    ipf.accelerator.print(x_ens_means[-1], x_ens_stds[-1])
                    ipf.accelerator.print("RMSE:", rmses[-1])
                    ipf.accelerator.print("Filter RMSE:", filter_rmses[-1])

                    plt.clf()
                    fig = plt.figure(figsize=(15, 6))
                    for i in range(3):
                        plt.subplot(1, 3, i+1)
                        plt.plot(np.arange(T_spinup, t+1), x[T_spinup:t+1, i], color="C0")
                        if xdim == ydim:
                            plt.plot(np.arange(T_spinup, t+1), y[T_spinup:t+1, i], 'o', color="C1")
                        plt.plot(np.arange(T_spinup, t+1), x_ens_means[T_spinup:, i], '--', color="C2")
                        plt.fill_between(np.arange(T_spinup, t+1),
                                         x_ens_means[T_spinup:, i] - x_ens_stds[T_spinup:, i],
                                         x_ens_means[T_spinup:, i] + x_ens_stds[T_spinup:, i],
                                         alpha=0.2, color="C2")
                        plt.plot(np.arange(T_spinup, t+1), gt_means_np[T_spinup:t+1, i], ':', color="C3")
                        plt.fill_between(np.arange(T_spinup, t+1),
                                         gt_means_np[T_spinup:t+1, i] - gt_stds_np[T_spinup:t+1, i],
                                         gt_means_np[T_spinup:t+1, i] + gt_stds_np[T_spinup:t+1, i],
                                         alpha=0.2, color="C3")
                    plt.savefig("im/filter_mean_std.png")

                    plt.clf()
                    plt.plot(np.arange(T_spinup, t+1), rmses[T_spinup:], label=f'CDSB (RMSE {np.mean(rmses[T_spinup:])})')
                    plt.plot(np.arange(T_spinup, t+1), rmses_enkf[T_spinup:t+1], label=f'EnKF (RMSE {np.mean(rmses_enkf[T_spinup:t+1])})')
                    plt.legend()
                    plt.savefig("im/rmse.png")

                    plt.clf()
                    plt.plot(np.arange(T_spinup, t+1), filter_rmses[T_spinup:], label=f'CDSB (Filter RMSE {np.mean(filter_rmses[T_spinup:])})')
                    plt.plot(np.arange(T_spinup, t+1), filter_rmses_enkf[T_spinup:t+1], label=f'EnKF (Filter RMSE {np.mean(filter_rmses_enkf[T_spinup:t+1])})')
                    plt.legend()
                    plt.savefig("im/filter_rmse.png")

                    plt.clf()
                    plt.plot(np.arange(T_spinup, t+1), filter_std_rmses[T_spinup:], label=f'CDSB (Filter std RMSE {np.mean(filter_std_rmses[T_spinup:])})')
                    plt.plot(np.arange(T_spinup, t+1), filter_std_rmses_enkf[T_spinup:t+1], label=f'EnKF (Filter std RMSE {np.mean(filter_std_rmses_enkf[T_spinup:t+1])})')
                    plt.legend()
                    plt.savefig("im/filter_std_rmse.png")
                    plt.close()
                    
                    np.save("rmses_enkf.npy", rmses_enkf)
                    np.save("filter_rmses_enkf.npy", filter_rmses_enkf)
                    np.save("filter_std_rmses_enkf.npy", filter_std_rmses_enkf)
                    
                    np.save("x_ens_means.npy", x_ens_means)
                    np.save("x_ens_stds.npy", x_ens_stds)
                    np.save("rmses.npy", rmses)
                    np.save("filter_rmses.npy", filter_rmses)
                    np.save("filter_std_rmses.npy", filter_std_rmses)

                if args.cond_final:
                    if args.update_cond_final_ens:
                        EnKF.x_T = x_ens
                    else:
                        x_ens = EnKF.x_T

                ipf.accelerator.free_memory()
                del ipf

        mean_rmse = np.mean(rmses[T_spinup:])
        mean_filter_rmse = np.mean(filter_rmses[T_spinup:])
        print("Mean RMSE:", mean_rmse)
        print("Mean filter RMSE:", mean_filter_rmse)


if __name__ == '__main__':
    main()

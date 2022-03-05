import os
import numpy as np

base_dir = "experiments/lorenz_type1/2022-02-07/cfg-cond_final=True,ens_size=2000,gamma_max=0.1,model.deg=4,num_steps=20/14-41-12"

n_exp = 10
rmses = []
filter_rmses = []
filter_std_rmses = []
rmses_enkf = []
filter_rmses_enkf = []
filter_std_rmses_enkf = []

for i in range(n_exp):
    exp_dir = os.path.join(base_dir, str(i))
    rmses.append(np.mean(np.load(os.path.join(exp_dir, "rmses.npy"))[2000:]))
    filter_rmses.append(np.mean(np.load(os.path.join(exp_dir, "filter_rmses.npy"))[2000:]))
    filter_std_rmses.append(np.mean(np.load(os.path.join(exp_dir, "filter_std_rmses.npy"))[2000:]))
    # rmses_enkf.append(np.mean(np.load(os.path.join(exp_dir, "rmses_enkf.npy"))[2000:]))
    # filter_rmses_enkf.append(np.mean(np.load(os.path.join(exp_dir, "filter_rmses_enkf.npy"))[2000:]))
    # filter_std_rmses_enkf.append(np.mean(np.load(os.path.join(exp_dir, "filter_std_rmses_enkf.npy"))[2000:]))

print("RMSE:", np.mean(rmses), np.std(rmses))
print("Filter RMSE:", np.mean(filter_rmses), np.std(filter_rmses))
print("Filter std RMSE:", np.mean(filter_std_rmses), np.std(filter_std_rmses))
# print("RMSE (EnKF):", np.mean(rmses_enkf), np.std(rmses_enkf))
# print("Filter RMSE (EnKF):", np.mean(filter_rmses_enkf), np.std(filter_rmses_enkf))
# print("Filter std RMSE (EnKF):", np.mean(filter_std_rmses_enkf), np.std(filter_std_rmses_enkf))

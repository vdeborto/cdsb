import os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib import pyplot as plt

results = ["experiments/biochemical_type1/2022-02-14/cfg-gamma_max=0.01,gamma_min=0.01,n_ipf=100/10-51-05/statistics_mean.npz",
           "experiments/biochemical_type1/2022-02-14/cfg-gamma_max=0.01,gamma_min=0.01,n_ipf=100/10-51-05/statistics_mean_fwdbwd.npz",
           "experiments/biochemical_type1/2022-02-14/cfg-cond_final=True,cond_final_model.adaptive_std=False,n_ipf=100/11-33-12/statistics_mean.npz"]
result_names = ["CDSB", "CDSB-FB", "CDSB-Cond"]

N = 100

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5))
for i in range(len(results)):
    result = results[i]
    result_name = result_names[i]

    result_npz = np.load(result)
    x0_means_mean, x1_means_mean, x0_vars_mean, x1_vars_mean, x0_skews_mean, x1_skews_mean, x0_kurts_mean, x1_kurts_mean = [result_npz[f] for f in result_npz.files]

    ax1.plot(np.arange(1, N+1), np.sqrt(1/2 * ((x0_means_mean - 0.075)**2 + (x1_means_mean - 0.875)**2)), label=result_name)
    ax2.plot(np.arange(1, N+1), np.sqrt(1/2 * ((x0_vars_mean - 0.190)**2 + (x1_vars_mean - 0.397)**2)), label=result_name)
    ax3.plot(np.arange(1, N+1), np.sqrt(1/2 * ((x0_skews_mean - 1.935)**2 + (x1_skews_mean - 0.681)**2)), label=result_name)
    ax4.plot(np.arange(1, N+1), np.sqrt(1/2 * ((x0_kurts_mean - 8.537)**2 + (x1_kurts_mean - 3.437)**2)), label=result_name)

ax1.set_title("Mean")
ax2.set_title("Variance")
ax3.set_title("Skew")
ax4.set_title("Kurtosis")

ax1.set_xlabel("CDSB Iteration")
ax2.set_xlabel("CDSB Iteration")
ax3.set_xlabel("CDSB Iteration")
ax4.set_xlabel("CDSB Iteration")

ax4.legend()
plt.tight_layout()
plt.savefig("biochemical_result.png")


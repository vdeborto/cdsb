import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_dir = "experiments/biochemical_type1/2022-02-11/cfg-n_ipf=100/15-05-02"
cond_final = False

n_exp = 1
x0_means = []
x1_means = []
x0_vars = []
x1_vars = []
x0_skews = []
x1_skews = []
x0_kurts = []
x1_kurts = []
if not cond_final:
    x0_means_fwdbwd = []
    x1_means_fwdbwd = []
    x0_vars_fwdbwd = []
    x1_vars_fwdbwd = []
    x0_skews_fwdbwd = []
    x1_skews_fwdbwd = []
    x0_kurts_fwdbwd = []
    x1_kurts_fwdbwd = []

min_len = np.inf

for i in range(n_exp):
    exp_dir = os.path.join(base_dir, str(i))
    df = pd.read_csv(os.path.join(exp_dir, "test_logs/version_0/metrics.csv"))
    df_b = df.loc[(df['fb'] == 'b')]

    x0_means.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_mean_'].to_numpy())
    x1_means.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_mean_'].to_numpy())
    x0_vars.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_var_'].to_numpy())
    x1_vars.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_var_'].to_numpy())
    x0_skews.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_skew_'].to_numpy())
    x1_skews.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_skew_'].to_numpy())
    x0_kurts.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_kurt_'].to_numpy())
    x1_kurts.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_kurt_'].to_numpy())

    if not cond_final:
        x0_means_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_mean_fwdbwd'].to_numpy())
        x1_means_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_mean_fwdbwd'].to_numpy())
        x0_vars_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_var_fwdbwd'].to_numpy())
        x1_vars_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_var_fwdbwd'].to_numpy())
        x0_skews_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_skew_fwdbwd'].to_numpy())
        x1_skews_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_skew_fwdbwd'].to_numpy())
        x0_kurts_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x0_kurt_fwdbwd'].to_numpy())
        x1_kurts_fwdbwd.append(df_b.loc[(df_b['step'] > 1)]['cond/x1_kurt_fwdbwd'].to_numpy())

    min_len = min(min_len, len(x0_means[-1]))

for i in range(n_exp):
    x0_means[i] = x0_means[i][:min_len]
    x1_means[i] = x1_means[i][:min_len]
    x0_vars[i] = x0_vars[i][:min_len]
    x1_vars[i] = x1_vars[i][:min_len]
    x0_skews[i] = x0_skews[i][:min_len]
    x1_skews[i] = x1_skews[i][:min_len]
    x0_kurts[i] = x0_kurts[i][:min_len]
    x1_kurts[i] = x1_kurts[i][:min_len]

    if not cond_final:
        x0_means_fwdbwd[i] = x0_means_fwdbwd[i][:min_len]
        x1_means_fwdbwd[i] = x1_means_fwdbwd[i][:min_len]
        x0_vars_fwdbwd[i] = x0_vars_fwdbwd[i][:min_len]
        x1_vars_fwdbwd[i] = x1_vars_fwdbwd[i][:min_len]
        x0_skews_fwdbwd[i] = x0_skews_fwdbwd[i][:min_len]
        x1_skews_fwdbwd[i] = x1_skews_fwdbwd[i][:min_len]
        x0_kurts_fwdbwd[i] = x0_kurts_fwdbwd[i][:min_len]
        x1_kurts_fwdbwd[i] = x1_kurts_fwdbwd[i][:min_len]

x0_means_mean = np.mean(x0_means, axis=0)
x1_means_mean = np.mean(x1_means, axis=0)
x0_vars_mean = np.mean(x0_vars, axis=0)
x1_vars_mean = np.mean(x1_vars, axis=0)
x0_skews_mean = np.mean(x0_skews, axis=0)
x1_skews_mean = np.mean(x1_skews, axis=0)
x0_kurts_mean = np.mean(x0_kurts, axis=0)
x1_kurts_mean = np.mean(x1_kurts, axis=0)

np.savez(os.path.join(base_dir, "statistics_mean.npz"), x0_means_mean, x1_means_mean, x0_vars_mean, x1_vars_mean,
         x0_skews_mean, x1_skews_mean, x0_kurts_mean, x1_kurts_mean)

if not cond_final:
    x0_means_fwdbwd_mean = np.mean(x0_means_fwdbwd, axis=0)
    x1_means_fwdbwd_mean = np.mean(x1_means_fwdbwd, axis=0)
    x0_vars_fwdbwd_mean = np.mean(x0_vars_fwdbwd, axis=0)
    x1_vars_fwdbwd_mean = np.mean(x1_vars_fwdbwd, axis=0)
    x0_skews_fwdbwd_mean = np.mean(x0_skews_fwdbwd, axis=0)
    x1_skews_fwdbwd_mean = np.mean(x1_skews_fwdbwd, axis=0)
    x0_kurts_fwdbwd_mean = np.mean(x0_kurts_fwdbwd, axis=0)
    x1_kurts_fwdbwd_mean = np.mean(x1_kurts_fwdbwd, axis=0)

    np.savez(os.path.join(base_dir, "statistics_mean_fwdbwd.npz"), x0_means_fwdbwd_mean, x1_means_fwdbwd_mean, x0_vars_fwdbwd_mean, x1_vars_fwdbwd_mean,
             x0_skews_fwdbwd_mean, x1_skews_fwdbwd_mean, x0_kurts_fwdbwd_mean, x1_kurts_fwdbwd_mean)

# print(x0_means_mean[-1], np.std(x0_means, axis=0)[-1])
# print(x1_means_mean[-1], np.std(x1_means, axis=0)[-1])
# print(x0_vars_mean[-1], np.std(x0_vars, axis=0)[-1])
# print(x1_vars_mean[-1], np.std(x1_vars, axis=0)[-1])
# print(x0_skews_mean[-1], np.std(x0_skews, axis=0)[-1])
# print(x1_skews_mean[-1], np.std(x1_skews, axis=0)[-1])
# print(x0_kurts_mean[-1], np.std(x0_kurts, axis=0)[-1])
# print(x1_kurts_mean[-1], np.std(x1_kurts, axis=0)[-1])

print(np.mean(x0_means_mean[-10:]))
print(np.mean(x1_means_mean[-10:]))
print(np.mean(x0_vars_mean[-10:]))
print(np.mean(x1_vars_mean[-10:]))
print(np.mean(x0_skews_mean[-10:]))
print(np.mean(x1_skews_mean[-10:]))
print(np.mean(x0_kurts_mean[-10:]))
print(np.mean(x1_kurts_mean[-10:]))

if not cond_final:
    print("Forward-Backward: ")
    print(np.mean(x0_means_fwdbwd_mean[-10:]))
    print(np.mean(x1_means_fwdbwd_mean[-10:]))
    print(np.mean(x0_vars_fwdbwd_mean[-10:]))
    print(np.mean(x1_vars_fwdbwd_mean[-10:]))
    print(np.mean(x0_skews_fwdbwd_mean[-10:]))
    print(np.mean(x1_skews_fwdbwd_mean[-10:]))
    print(np.mean(x0_kurts_fwdbwd_mean[-10:]))
    print(np.mean(x1_kurts_fwdbwd_mean[-10:]))

# Mean
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_means_mean))+1, x0_means_mean, label="x1", color="C0")
if not cond_final:
    ax1.plot(np.arange(len(x0_means_fwdbwd_mean))+1, x0_means_fwdbwd_mean, label="x1", color="C0", linestyle="--")
ax1.axhline(0.075, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_means_mean))+1, x1_means_mean, label="x2", color="C1")
if not cond_final:
    ax2.plot(np.arange(len(x1_means_fwdbwd_mean))+1, x1_means_fwdbwd_mean, label="x2", color="C1", linestyle="--")
ax2.axhline(0.875, ls='-.', color="C1")
# plt.legend(handles=[line1, line2])
plt.title("Posterior Mean")
plt.savefig(os.path.join(base_dir, "mean.png"))


# Var
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_vars_mean))+1, x0_vars_mean, label="x1", color="C0")
if not cond_final:
    ax1.plot(np.arange(len(x0_vars_fwdbwd_mean))+1, x0_vars_fwdbwd_mean, label="x1", color="C0", linestyle="--")
ax1.axhline(0.190, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_vars_mean))+1, x1_vars_mean, label="x2", color="C1")
if not cond_final:
    ax2.plot(np.arange(len(x1_vars_fwdbwd_mean))+1, x1_vars_fwdbwd_mean, label="x2", color="C1", linestyle="--")
ax2.axhline(0.397, ls='-.', color="C1")
# plt.legend(handles=[line1, line2])
plt.title("Posterior Var")
plt.savefig(os.path.join(base_dir, "var.png"))


# Skew
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_skews_mean))+1, x0_skews_mean, label="x1", color="C0")
if not cond_final:
    ax1.plot(np.arange(len(x0_skews_fwdbwd_mean))+1, x0_skews_fwdbwd_mean, label="x1", color="C0", linestyle="--")
ax1.axhline(1.935, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_skews_mean))+1, x1_skews_mean, label="x2", color="C1")
if not cond_final:
    ax2.plot(np.arange(len(x1_skews_fwdbwd_mean))+1, x1_skews_fwdbwd_mean, label="x2", color="C1", linestyle="--")
ax2.axhline(0.681, ls='-.', color="C1")
# plt.legend(handles=[line1, line2])
plt.title("Posterior Skew")
plt.savefig(os.path.join(base_dir, "skew.png"))


# Kurtosis
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_kurts_mean))+1, x0_kurts_mean, label="x1", color="C0")
if not cond_final:
    ax1.plot(np.arange(len(x0_kurts_fwdbwd_mean))+1, x0_kurts_fwdbwd_mean, label="x1", color="C0", linestyle="--")
ax1.axhline(8.537, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_kurts_mean))+1, x1_kurts_mean, label="x2", color="C1")
if not cond_final:
    ax2.plot(np.arange(len(x1_kurts_fwdbwd_mean))+1, x1_kurts_fwdbwd_mean, label="x2", color="C1", linestyle="--")
ax2.axhline(3.437, ls='-.', color="C1")
plt.legend(handles=[line1, line2])
plt.title("Posterior Kurtosis")
plt.savefig(os.path.join(base_dir, "kurt.png"))
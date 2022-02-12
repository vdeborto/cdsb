import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_dir = "experiments/biochemical_type1/2022-02-11/cfg-n_ipf=100/15-05-02"

n_exp = 10
x0_means = []
x1_means = []
x0_vars = []
x1_vars = []
x0_skews = []
x1_skews = []
x0_kurts = []
x1_kurts = []

min_len = np.inf

for i in range(n_exp):
    exp_dir = os.path.join(base_dir, str(i))
    df = pd.read_csv(os.path.join(exp_dir, "test_logs/version_0/metrics.csv"))
    df_b = df.loc[(df['fb'] == 'b')]

    x0_means.append(df_b['cond/x0_mean_'].to_numpy()[1:])
    x1_means.append(df_b['cond/x1_mean_'].to_numpy()[1:])
    x0_vars.append(df_b['cond/x0_var_'].to_numpy()[1:])
    x1_vars.append(df_b['cond/x1_var_'].to_numpy()[1:])
    x0_skews.append(df_b['cond/x0_skew_'].to_numpy()[1:])
    x1_skews.append(df_b['cond/x1_skew_'].to_numpy()[1:])
    x0_kurts.append(df_b['cond/x0_kurt_'].to_numpy()[1:])
    x1_kurts.append(df_b['cond/x1_kurt_'].to_numpy()[1:])

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

x0_means_mean = np.mean(x0_means, axis=0)
x1_means_mean = np.mean(x1_means, axis=0)
x0_vars_mean = np.mean(x0_vars, axis=0)
x1_vars_mean = np.mean(x1_vars, axis=0)
x0_skews_mean = np.mean(x0_skews, axis=0)
x1_skews_mean = np.mean(x1_skews, axis=0)
x0_kurts_mean = np.mean(x0_kurts, axis=0)
x1_kurts_mean = np.mean(x1_kurts, axis=0)

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

# Mean
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_means_mean))+1, x0_means_mean, label="x1", color="C0")
ax1.axhline(0.049, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_means_mean))+1, x1_means_mean, label="x2", color="C1")
ax2.axhline(0.924, ls='-.', color="C1")
# plt.legend(handles=[line1, line2])
plt.title("Posterior Mean")
plt.savefig(os.path.join(base_dir, "mean.png"))


# Var
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_vars_mean))+1, x0_vars_mean, label="x1", color="C0")
ax1.axhline(0.176, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_vars_mean))+1, x1_vars_mean, label="x2", color="C1")
ax2.axhline(0.406, ls='-.', color="C1")
# plt.legend(handles=[line1, line2])
plt.title("Posterior Var")
plt.savefig(os.path.join(base_dir, "var.png"))


# Skew
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_skews_mean))+1, x0_skews_mean, label="x1", color="C0")
ax1.axhline(2.03, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_skews_mean))+1, x1_skews_mean, label="x2", color="C1")
ax2.axhline(0.643, ls='-.', color="C1")
# plt.legend(handles=[line1, line2])
plt.title("Posterior Skew")
plt.savefig(os.path.join(base_dir, "skew.png"))


# Kurtosis
fig, ax1 = plt.subplots()
line1, = ax1.plot(np.arange(len(x0_kurts_mean))+1, x0_kurts_mean, label="x1", color="C0")
ax1.axhline(8.97, ls='-.', color="C0")
ax1.set_xlabel("IPF Iteration")
ax2 = ax1.twinx()
line2, = ax2.plot(np.arange(len(x1_kurts_mean))+1, x1_kurts_mean, label="x2", color="C1")
ax2.axhline(3.38, ls='-.', color="C1")
plt.legend(handles=[line1, line2])
plt.title("Posterior Kurtosis")
plt.savefig(os.path.join(base_dir, "kurt.png"))
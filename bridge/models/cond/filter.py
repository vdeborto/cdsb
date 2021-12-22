import torch
import torch.nn as nn
from utils import sample_cov


class EnsembleKalmanFilter(nn.Module):
    def __init__(self, xdim, ydim, F_fn, G_fn, p_0_dist, ensemble_size):
        super().__init__()
        self.T = -1

        self.xdim = xdim
        self.ydim = ydim
        self.ensemble_size = ensemble_size

        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist
        self.x_Tm1 = None
        self.x_T = None

    def advance_timestep(self, y_T):
        self.T += 1

        if self.T == 0:
            self.x_T_pred = self.p_0_dist().sample((self.ensemble_size,))
        else:
            self.x_Tm1 = self.x_T.clone().detach()
            self.x_T_pred = self.F_fn(self.x_Tm1, self.T-1).sample()

        self.y_T_pred = self.G_fn(self.x_T_pred, self.T).sample()
        cov_x_y = sample_cov(self.x_T_pred, self.y_T_pred)
        cov_y = sample_cov(self.y_T_pred)
        self.K = cov_x_y @ torch.linalg.inv(cov_y)

    def update(self, y_T):
        self.x_T = self.x_T_pred + (y_T - self.y_T_pred) @ self.K.t()

        # if self.T > 0:
        #     cov_x_Tm1_y = sample_cov(self.x_Tm1, self.y_T_pred)
        #     J = cov_x_Tm1_y @ torch.linalg.inv(cov_y)
        #     self.x_Tm1 = self.x_Tm1 + (y_T - self.y_T_pred) @ J.t()

    def return_summary_stats(self, t=None):
        if t is None:
            t = self.T
        if t == self.T:
            x_t = self.x_T
        # elif t == self.T - 1:
        #     x_t = self.x_Tm1
        x_t_mean = x_t.mean(0)
        x_t_cov = sample_cov(x_t)
        return x_t_mean, x_t_cov

    def forward(self, y):
        x = self.x_T_pred.to(y.device) + (y.unsqueeze(-2) - self.y_T_pred.to(y.device)) @ self.K.to(y.device).t()
        return x.mean(-2), x.std(-2)

# class BootstrapParticleFilter(nn.Module):
#     def __init__(self, device, xdim, ydim, F_fn, G_fn, p_0_dist, num_particles):
#         """
#             Cond q is the locally optimal proposal
#         """
#         super().__init__()
#         self.T = -1
#
#         self.device = device
#         self.xdim = xdim
#         self.ydim = ydim
#         self.num_particles = num_particles
#
#         self.F_fn = F_fn
#         self.G_fn = G_fn
#         self.p_0_dist = p_0_dist
#
#         self.x_Tm1 = None
#         self.x_T = None
#
#         # Initial weights (1/num_particles)
#         self.log_w = np.log(1 / self.num_particles) * torch.ones((self.num_particles, 1), device=self.device)
#
#         self.amortised = False
#
#     def advance_timestep(self, y_T):
#         self.T += 1
#         if self.T > 0:
#             self.x_Tm1 = self.x_T.clone().detach()
#
#     def sample_q_T(self, y_T, num_samples, detach_x=False):
#         if self.T == 0:
#             pred_dist = self.p_0_dist()
#             x_T = pred_dist.rsample((self.num_particles, ))
#
#         else:
#             pred_dist = self.F_fn(self.x_Tm1, self.T-1)
#             x_T = pred_dist.rsample()
#
#         if detach_x:
#             x_T = x_T.detach()
#
#         return x_T
#
#     def compute_log_p_t(self, x_t, y_t):  # Only need for time T
#         log_p_y_t = self.G_fn(x_t, self.T).log_prob(y_t).unsqueeze(1)
#         return {"log_p_y_t": log_p_y_t}
#
#     def update(self, y_T):
#         if self.resample_criterion():
#             self.resample()
#
#         self.x_T = self.sample_q_T(y_T, self.num_particles, detach_x=True)
#         log_p_T = self.compute_log_p_t(self.x_T, y_T)
#         self.log_w += log_p_T["log_p_y_t"]
#
#     def resample(self):
#         resampling_dist = Categorical(logits=self.log_w[:, 0])
#         ancestors = resampling_dist.sample((self.num_particles,))
#         self.x_Tm1 = self.x_Tm1[ancestors, :]
#         self.log_w = np.log(1 / self.num_particles) * torch.ones((self.num_particles, 1), device=self.device)
#
#     def resample_criterion(self):
#         if self.T > 0:
#             return utils.ess(self.log_w) <= self.num_particles / 2
#         else:
#             return False
#
#     def return_summary_stats(self, t=None):
#         if t is None:
#             t = self.T
#         normalized_w = functional.softmax(self.log_w, dim=0)
#         if t == self.T:
#             x_t = self.x_T
#         elif t == self.T - 1:
#             x_t = self.x_Tm1
#         x_t_mean = (x_t * normalized_w).sum(0)
#         x_t_cov = sample_cov(x_t, w=normalized_w)
#         return x_t_mean, x_t_cov
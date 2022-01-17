import time
import os, sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torchvision.utils as vutils
from . import repeater
from ..data.utils import save_image, to_uint8_tensor, normalize_tensor
from ..data.metrics import PSNR, SSIM, FID
from PIL import Image
# matplotlib.use('Agg')
from scipy.stats import kde, gamma, norm, lognorm


DPI = 200

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)


class Plotter(object):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif'):
        self.ipf = ipf
        self.args = args
        self.plot_level = self.args.plot_level

        self.dataset = self.args.data.dataset
        self.num_steps = self.args.num_steps

        if self.ipf.accelerator.is_main_process:
            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

        while not os.path.isdir(im_dir):
            time.sleep(3)
        ipf.accelerator.wait_for_everyone()

        existing_versions = []
        for d in os.listdir(im_dir):
            if os.path.isdir(os.path.join(im_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            version = 0
        else:
            version = max(existing_versions) + 1

        self.im_dir = os.path.join(im_dir, f"version_{version}")
        self.gif_dir = os.path.join(gif_dir, f"version_{version}")

        if self.ipf.accelerator.is_main_process:
            os.makedirs(self.im_dir, exist_ok=True)
            os.makedirs(self.gif_dir, exist_ok=True)

        self.metrics_dict = {}

    def __call__(self, sample_net, i, n, fb):
        out = {}
        self.step = max(i+self.ipf.num_iter*(n-1), 0)

        if self.ipf.accelerator.is_main_process:
            out['fb'] = fb
            out['ipf'] = n
            out['T'] = self.ipf.T

        for dl_name, dl in self.ipf.save_dls_dict.items():
            x_start, y_start, x_tot, x_init, mean_final, var_final, time_joint, metric_results = \
                self.generate_sequence_joint(dl, sample_net, i, n, fb, dl_name=dl_name)
            out.update(metric_results)
            out[dl_name + "/" + 'batch_sample_time'] = np.mean(time_joint)

            if self.ipf.accelerator.is_main_process:
                if self.args.cond_final:
                    self.plot_sequence_joint(x_start[:self.args.plot_npar], y_start[:self.args.plot_npar],
                                             x_tot[:, :self.args.plot_npar], x_init[:self.args.plot_npar],
                                             self.dataset, i, n, fb, dl_name=dl_name,
                                             mean_final=mean_final[:self.args.plot_npar],
                                             var_final=var_final[:self.args.plot_npar])
                    out.update(self.test_joint(x_start[:self.args.test_npar], y_start[:self.args.test_npar],
                                               x_tot[:, :self.args.test_npar], x_init[:self.args.test_npar],
                                               self.dataset, i, n, fb, dl_name=dl_name,
                                               mean_final=mean_final[:self.args.test_npar],
                                               var_final=var_final[:self.args.test_npar]))
                else:
                    self.plot_sequence_joint(x_start[:self.args.plot_npar], y_start[:self.args.plot_npar],
                                             x_tot[:, :self.args.plot_npar], x_init[:self.args.plot_npar],
                                             self.dataset, i, n, fb, dl_name=dl_name,
                                             mean_final=mean_final, var_final=var_final)
                    out.update(self.test_joint(x_start[:self.args.test_npar], y_start[:self.args.test_npar],
                                               x_tot[:, :self.args.test_npar], x_init[:self.args.test_npar],
                                               self.dataset, i, n, fb, dl_name=dl_name,
                                               mean_final=mean_final, var_final=var_final))

        if n > 0 and self.ipf.y_cond is not None:
            if fb == 'b':
                x_start_cond = []
                x_tot_cond = []
                time_cond = []
                save_init_dl = repeater(self.ipf.save_init_dl)
                for y_c in self.ipf.y_cond:
                    x_start, x_tot_c, time_c = self.generate_sequence_cond(y_c, save_init_dl, sample_net, i, n, fb)
                    if self.ipf.accelerator.is_main_process:
                        x_start_cond.append(x_start)
                        x_tot_cond.append(x_tot_c)
                        time_cond.append(time_c)

                if self.ipf.accelerator.is_main_process:
                    x_start_cond = torch.stack(x_start_cond, dim=0)
                    x_tot_cond = torch.stack(x_tot_cond, dim=0)
                    self.plot_sequence_cond(x_start_cond[:, :self.args.plot_npar], self.ipf.y_cond, x_tot_cond[:, :, :self.args.plot_npar],
                                            self.dataset, i, n, fb, x_init_cond=self.ipf.x_cond_true)
                    out.update(self.test_cond(x_start_cond[:, :self.args.test_npar], self.ipf.y_cond, x_tot_cond[:, :, :self.args.test_npar],
                                              self.dataset, i, n, fb, x_init_cond=self.ipf.x_cond_true))
                    out["cond/batch_sample_time"] = np.mean(time_cond)

            if not self.args.cond_final:
                x_init_cond = []
                y_init_cond = []
                x_tot_fwd_cond = []
                x_tot_fwdbwd_cond = []
                time_cond = []
                save_init_dl = repeater(self.ipf.save_init_dl)
                for y_c in self.ipf.y_cond:
                    x_init, y_init, x_tot_fwd, x_tot_fwdbwd_c, time_c = self.generate_sequence_cond_fwdbwd(y_c, save_init_dl, sample_net, i, n, fb)
                    if self.ipf.accelerator.is_main_process:
                        x_init_cond.append(x_init)
                        y_init_cond.append(y_init)
                        x_tot_fwd_cond.append(x_tot_fwd)
                        x_tot_fwdbwd_cond.append(x_tot_fwdbwd_c)
                        time_cond.append(time_c)

                if self.ipf.accelerator.is_main_process:
                    x_init_cond = torch.stack(x_init_cond, dim=0)
                    y_init_cond = torch.stack(y_init_cond, dim=0)
                    x_tot_fwd_cond = torch.stack(x_tot_fwd_cond, dim=0)
                    x_tot_fwdbwd_cond = torch.stack(x_tot_fwdbwd_cond, dim=0)
                    self.plot_sequence_cond_fwdbwd(x_init_cond[:, :self.args.plot_npar], y_init_cond[:, :self.args.plot_npar],
                                                   x_tot_fwd_cond[:, :, :self.args.plot_npar], self.ipf.y_cond, x_tot_fwdbwd_cond[:, :, :self.args.plot_npar],
                                                   self.dataset, i, n, fb, x_init_cond=self.ipf.x_cond_true)
                    out.update(self.test_cond(x_tot_fwd_cond[:, -1, :self.args.test_npar], self.ipf.y_cond, x_tot_fwdbwd_cond[:, :, :self.args.test_npar],
                                              self.dataset, i, n, fb, x_init_cond=self.ipf.x_cond_true))
                    out["cond/batch_sample_time_fwdbwd"] = np.mean(time_cond)

        torch.cuda.empty_cache()
        return out

    def generate_sequence_joint(self, dl, sample_net, i, n, fb, dl_name='train'):
        iter_dl = iter(dl)
        for metric in self.metrics_dict.values():
            metric.reset()

        all_batch_x = []
        all_batch_y = []
        all_x_tot = []
        all_init_batch_x = []
        all_mean_final = []
        all_var_final = []
        times = []
        metric_results = {}

        iters = 0
        while iters * self.ipf.test_batch_size < (self.ipf.save_npar if fb == 'b' else self.ipf.args.test_npar):
            try:
                start = time.time()

                batch_x, batch_y, init_batch_x, mean_final, var_final = self.ipf.sample_batch(iter_dl, self.ipf.save_final_dl, fb)

                if n == 0:
                    assert fb == 'f'
                    x_tot, _, _, _ = self.ipf.langevin.record_init_langevin(batch_x, batch_y, mean_final=mean_final, var_final=var_final)
                else:
                    x_tot, _, _, _ = self.ipf.langevin.record_langevin_seq(sample_net, batch_x, batch_y, fb, sample=True, var_final=var_final)

                stop = time.time()
                times.append(stop - start)

                x_last = x_tot[:, -1]
                self.plot_and_record_batch_joint(x_last, init_batch_x, iters, i, n, fb, dl_name=dl_name)

                if iters * self.ipf.test_batch_size < self.ipf.args.test_npar:
                    gather_batch_x = self.ipf.accelerator.gather(batch_x)
                    gather_batch_y = self.ipf.accelerator.gather(batch_y)
                    gather_x_tot = self.ipf.accelerator.gather(x_tot)
                    gather_init_batch_x = self.ipf.accelerator.gather(init_batch_x)

                    if self.args.cond_final:
                        gather_mean_final = self.ipf.accelerator.gather(mean_final)
                        gather_var_final = self.ipf.accelerator.gather(var_final)

                    if self.ipf.accelerator.is_main_process:
                            all_batch_x.append(gather_batch_x.cpu())
                            all_batch_y.append(gather_batch_y.cpu())
                            all_x_tot.append(gather_x_tot.cpu())
                            all_init_batch_x.append(gather_init_batch_x.cpu())
                            if self.args.cond_final:
                                all_mean_final.append(gather_mean_final.cpu())
                                all_var_final.append(gather_var_final.cpu())

                iters = iters + 1

            except StopIteration:
                break

        if self.ipf.accelerator.is_main_process:
            all_batch_x = torch.cat(all_batch_x, dim=0)
            all_batch_y = torch.cat(all_batch_y, dim=0)
            all_x_tot = torch.cat(all_x_tot, dim=0)
            all_init_batch_x = torch.cat(all_init_batch_x, dim=0)

            shape_len = len(all_x_tot.shape)
            all_x_tot = all_x_tot.permute(1, 0, *list(range(2, shape_len)))

            if self.args.cond_final:
                all_mean_final = torch.cat(all_mean_final, dim=0)
                all_var_final = torch.cat(all_var_final, dim=0)
            else:
                all_mean_final = self.ipf.mean_final.cpu()
                all_var_final = self.ipf.var_final.cpu()

        if fb == 'b':
            for metric_name, metric in self.metrics_dict.items():
                metric_result = metric.compute()
                if self.ipf.accelerator.is_main_process:
                    metric_results[dl_name + "/" + metric_name] = metric_result
                metric.reset()

        return all_batch_x, all_batch_y, all_x_tot, all_init_batch_x, all_mean_final, all_var_final, times, metric_results

    def generate_sequence_cond(self, y_c, dl, sample_net, i, n, fb):
        if y_c is not None and fb == 'b':
            start = time.time()

            batch_x, _, _, _, var_final = self.ipf.sample_batch(dl, self.ipf.save_final_dl, fb, y_c=y_c)
            x_tot_c = self.ipf.backward_sample(batch_x, y_c, sample_net=sample_net, var_final=var_final)

            gather_batch_x = self.ipf.accelerator.gather(batch_x).cpu()
            gather_x_tot_c = self.ipf.accelerator.gather(x_tot_c).cpu()

            stop = time.time()
            return gather_batch_x, gather_x_tot_c, stop-start

    def generate_sequence_cond_fwdbwd(self, y_c, dl, sample_net, i, n, fb):
        if y_c is not None and not self.args.cond_final:
            start = time.time()

            batch_x, batch_y, init_batch_x, _, _ = self.ipf.sample_batch(dl, self.ipf.save_final_dl, fb)
            init_batch_y = batch_y

            if fb == 'f':
                x_tot_fwd, x_tot_fwdbwd_c = self.ipf.forward_backward_sample(init_batch_x, init_batch_y, y_c, n, fb,
                                                                             return_fwd_tot=True, sample_net_f=sample_net)
            elif fb == 'b':
                x_tot_fwd, x_tot_fwdbwd_c = self.ipf.forward_backward_sample(init_batch_x, init_batch_y, y_c, n, fb,
                                                                             return_fwd_tot=True, sample_net_b=sample_net)

            gather_init_batch_x = self.ipf.accelerator.gather(init_batch_x).cpu()
            gather_init_batch_y = self.ipf.accelerator.gather(init_batch_y).cpu()
            gather_x_tot_fwd = self.ipf.accelerator.gather(x_tot_fwd).cpu()
            gather_x_tot_fwdbwd_c = self.ipf.accelerator.gather(x_tot_fwdbwd_c).cpu()

            stop = time.time()
            return gather_init_batch_x, gather_init_batch_y, gather_x_tot_fwd, gather_x_tot_fwdbwd_c, stop-start

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', freq=None,
                            mean_final=None, var_final=None):
        if dl_name == 'train':
            if freq is None:
                freq = self.num_steps // min(self.num_steps, 50)
            iter_name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, iter_name, dl_name)
            gif_dir = os.path.join(self.gif_dir, dl_name)

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            x_tot = x_tot.cpu().reshape(x_tot.shape[0], x_tot.shape[1], -1).numpy()
            x_start = x_start.cpu().numpy()
            if mean_final is not None:
                mean_final = mean_final.cpu().numpy() + np.zeros_like(x_start[:1])
            if var_final is not None:
                var_final = var_final.cpu().numpy() + np.zeros_like(x_start[:1])
            x_start_tot = np.concatenate([np.expand_dims(x_start.reshape([x_start.shape[0], -1]), axis=0), x_tot], axis=0)

            if self.plot_level >= 5:
                plot_name = 'histogram'
                name_gif = f'{iter_name}_{plot_name}'
                plot_paths_reg = []
                dims = np.sort(np.random.choice(x_tot.shape[-1], min(x_tot.shape[-1], 3), replace=False))
                for k in range(self.num_steps+1):
                    if k % freq == 0 or k == self.num_steps:
                        filename = plot_name + '_' + str(k) + '.png'
                        filename = os.path.join(im_dir, filename)
                        plt.clf()
                        fig = plt.figure(figsize=(5*len(dims), 4))
                        if n is not None:
                            str_title = 'IPFP iteration: ' + str(n)
                            plt.title(str_title)

                        for d in range(len(dims)):
                            dim = dims[d]
                            x_min, x_max = np.min(x_start_tot[:, :, dim]), np.max(x_start_tot[:, :, dim])
                            ax = plt.subplot(1, len(dims), d+1)
                            plt.hist(x_start_tot[k, :, dim], bins=50, density=True, alpha=0.5)
                            if mean_final is not None and var_final is not None and fb == 'f' and not self.args.cond_final:
                                mu = mean_final.reshape([-1])[dim]
                                sig = np.sqrt(var_final.reshape([-1])[dim])
                                x_lin = np.linspace(x_min, x_max, 250)
                                plt.plot(x_lin, stats.norm.pdf(x_lin, mu, sig))
                            ax.set_xlim(x_min, x_max)
                        plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                        plt.close()
                        plot_paths_reg.append(filename)

                make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

            if self.plot_level >= 6 and x_start[0].shape == y_start[0].shape:
                npts = 250
                y_start = y_start.cpu().numpy().reshape(y_start.shape[0], -1)

                if n == 0 and fb == "f":
                    plt.clf()
                    filename = f'true_density_{dl_name}.png'
                    filename = os.path.join(self.im_dir, filename)
                    fig = plt.figure(figsize=(5 * len(dims), 4))
                    for d in range(len(dims)):
                        dim = dims[d]
                        x_min, x_max = np.min(x_start[:, dim]), np.max(x_start[:, dim])
                        y_min, y_max = np.min(y_start[:, dim]), np.max(y_start[:, dim])
                        ax = plt.subplot(1, len(dims), d+1)
                        g = sns.kdeplot(ax=ax, x=x_start[:, dim], y=y_start[:, dim])
                        g.set(xlim=(x_min, x_max))
                        g.set(ylim=(y_min, y_max))
                        plt.xlabel("x")
                        plt.ylabel("y")
                    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                    plt.close()

                plot_name = 'density'
                name_gif = f'{iter_name}_{plot_name}'
                plot_paths_reg = []
                for k in range(self.num_steps+1):
                    if k % freq == 0 or k == self.num_steps:
                        filename = plot_name + '_' + str(k) + '.png'
                        filename = os.path.join(im_dir, filename)
                        plt.clf()
                        fig = plt.figure(figsize=(5*len(dims), 4))
                        if n is not None:
                            str_title = 'IPFP iteration: ' + str(n)
                            plt.title(str_title)

                        for d in range(len(dims)):
                            dim = dims[d]
                            x_min, x_max = np.min(x_start_tot[:, :, dim]), np.max(x_start_tot[:, :, dim])
                            y_min, y_max = np.min(y_start[:, dim]), np.max(y_start[:, dim])
                            ax = plt.subplot(1, len(dims), d+1)
                            g = sns.kdeplot(ax=ax, x=x_start_tot[k, :, dim], y=y_start[:, dim])
                            g.set(xlim=(x_min, x_max))
                            g.set(ylim=(y_min, y_max))
                            plt.xlabel("x")
                            plt.ylabel("y")
                        plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                        plt.close()
                        plot_paths_reg.append(filename)

                make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        pass

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        pass

    def test_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', mean_final=None, var_final=None):
        out = {}
        if dl_name == 'train':
            x_last = x_tot[-1]
    
            x_var_last = torch.var(x_last, dim=0).mean().item()
            x_var_start = torch.var(x_start, dim=0).mean().item()
            x_mean_last = torch.mean(x_last).item()
            x_mean_start = torch.mean(x_start).item()
    
            out = {'x_mean_start': x_mean_start, 'x_var_start': x_var_start,
                   'x_mean_last': x_mean_last, 'x_var_last': x_var_last}
    
            if mean_final is not None:
                x_mse_last = torch.mean((x_last - mean_final) ** 2)
                x_mse_start = torch.mean((x_start - mean_final) ** 2)
                out.update({"x_mse_start": x_mse_start, "x_mse_last": x_mse_last})
    
        return out

    def test_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag=''):
        return {}

    def plot_and_record_batch_joint(self, x_last, x_init, iters, i, n, fb, dl_name='train'):
        pass


class ImPlotter(Plotter):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif'):
        super().__init__(ipf, args, im_dir=im_dir, gif_dir=gif_dir)
        self.num_plots_grid = 100

        self.metrics_dict = {"psnr": PSNR(data_range=255.).to(self.ipf.device),
                             "ssim": SSIM(data_range=255.).to(self.ipf.device),
                             "fid": FID().to(self.ipf.device)}

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', freq=None,
                            mean_final=None, var_final=None):
        super().plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, freq=freq,
                                    mean_final=mean_final, var_final=var_final)
        if freq is None:
            freq = self.num_steps // min(self.num_steps, 50)

        if self.plot_level >= 1:
            x_tot_grid = x_tot[:, :self.num_plots_grid]
            name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, name, dl_name)
            gif_dir = os.path.join(self.gif_dir, dl_name)

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            # plot_level 1
            uint8_x_init = to_uint8_tensor(x_init[:self.num_plots_grid])

            def save_image_with_metrics(batch_x, filename, **kwargs):
                plt.clf()

                uint8_batch_x = to_uint8_tensor(batch_x)

                psnr = PSNR(data_range=255.)
                psnr_result = psnr(uint8_batch_x, uint8_x_init).item()
                psnr.reset()

                ssim = SSIM(data_range=255.)
                ssim_result = ssim(uint8_batch_x, uint8_x_init).item()
                ssim.reset()

                uint8_batch_x_grid = vutils.make_grid(uint8_batch_x, **kwargs).permute(1, 2, 0)
                plt.imshow(uint8_batch_x_grid)

                plt.title('IPFP iteration: ' + str(n) + '\n psnr: ' + str(round(psnr_result, 2)) + '\n ssim: ' + str(
                    round(ssim_result, 2)))
                plt.axis('off')
                plt.savefig(filename)
                plt.close()

            filename_grid_png = os.path.join(im_dir, 'im_grid_start.png')
            save_image_with_metrics(x_start[:self.num_plots_grid], filename_grid_png, nrow=10)
            self.ipf.save_logger.log_image(dl_name + "/im_grid_start", [filename_grid_png], step=self.step, fb=fb)

            filename_grid_png = os.path.join(im_dir, 'im_grid_last.png')
            save_image_with_metrics(x_tot_grid[-1], filename_grid_png, nrow=10)
            self.ipf.save_logger.log_image(dl_name + "/im_grid_last", [filename_grid_png], step=self.step, fb=fb)

            filename_grid_png = os.path.join(im_dir, 'im_grid_data_x.png')
            save_image_with_metrics(x_init[:self.num_plots_grid], filename_grid_png, nrow=10)
            self.ipf.save_logger.log_image(dl_name + "/im_grid_data_x", [filename_grid_png], step=self.step, fb=fb)

            filename_grid_png = os.path.join(im_dir, 'im_grid_data_y.png')
            save_image_with_metrics(y_start[:self.num_plots_grid], filename_grid_png, nrow=10)
            self.ipf.save_logger.log_image(dl_name + "/im_grid_data_y", [filename_grid_png], step=self.step, fb=fb)

            if self.plot_level >= 2:
                plot_paths = []
                x_start_tot_grid = torch.cat([x_start[:self.num_plots_grid].unsqueeze(0), x_tot_grid], dim=0)
                for k in range(self.num_steps+1):
                    if k % freq == 0 or k == self.num_steps:
                        # save png
                        filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))
                        plot_paths.append(filename_grid_png)
                        save_image_with_metrics(x_start_tot_grid[k], filename_grid_png, nrow=10)

                make_gif(plot_paths, output_directory=gif_dir, gif_name=name+'_im_grid')

    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        if freq is None:
            freq = self.num_steps // min(self.num_steps, 50)

        if self.plot_level >= 1:
            y_cond = y_cond.cpu()
            x_init_cond = x_init_cond.cpu() if x_init_cond is not None else None

            iter_name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, iter_name, "cond")
            gif_dir = os.path.join(self.gif_dir, "cond")

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            # plot_level 1
            if x_init_cond is not None:
                uint8_x_init = to_uint8_tensor(x_init_cond[:self.num_plots_grid])

            def save_image_with_metrics(batch_x, filename):
                # batch_x shape (y_cond, b, c, h, w)
                plt.clf()
                ncol = 10 if x_init_cond is not None else 9
                plt.figure(figsize=(ncol, batch_x.shape[0]))

                uint8_batch_x = to_uint8_tensor(batch_x)
                batch_x_mean = batch_x.mean(1)
                batch_x_std = batch_x.std(1)
                uint8_batch_x_mean = to_uint8_tensor(batch_x_mean)

                plt_idx = 1

                def subplot_imshow(tensor, plt_idx):
                    ax = plt.subplot(len(y_cond), ncol, plt_idx)
                    ax.axis('off')
                    ax.imshow(tensor.permute(1, 2, 0))

                for j in range(len(y_cond)):
                    if x_init_cond is not None:
                        subplot_imshow(uint8_x_init[j].expand(3, -1, -1), plt_idx)
                        plt_idx += 1
                    subplot_imshow(to_uint8_tensor(y_cond[j]).expand(3, -1, -1), plt_idx)
                    plt_idx += 1
                    for k in range(6):
                        subplot_imshow(uint8_batch_x[j, k].expand(3, -1, -1), plt_idx)
                        plt_idx += 1
                    if batch_x_mean.shape[1] == 1:
                        subplot_imshow(batch_x_mean[j], plt_idx)
                    elif batch_x_mean.shape[1] == 3:
                        subplot_imshow(uint8_batch_x_mean[j], plt_idx)
                    else:
                        raise ValueError
                    plt_idx += 1
                    subplot_imshow(batch_x_std[j], plt_idx)
                    plt_idx += 1

                psnr = PSNR(data_range=255.)
                psnr_result = psnr(uint8_batch_x.flatten(end_dim=1),
                                   uint8_x_init.unsqueeze(1).expand(-1, batch_x.shape[1], -1, -1, -1).flatten(end_dim=1)).item()
                psnr.reset()
                mean_psnr_result = psnr(uint8_batch_x_mean, uint8_x_init).item()
                psnr.reset()

                ssim = SSIM(data_range=255.)
                ssim_result = ssim(uint8_batch_x.flatten(end_dim=1),
                                   uint8_x_init.unsqueeze(1).expand(-1, batch_x.shape[1], -1, -1, -1).flatten(end_dim=1)).item()
                ssim.reset()
                mean_ssim_result = ssim(uint8_batch_x_mean, uint8_x_init).item()
                ssim.reset()

                plt.suptitle('IPFP iteration: ' + str(n) +
                             '\n psnr: ' + str(round(psnr_result, 2)) + ' mean psnr: ' + str(round(mean_psnr_result, 2)) +
                             '\n ssim: ' + str(round(ssim_result, 2)) + ' mean ssim: ' + str(round(mean_ssim_result, 2)))
                plt.savefig(filename)
                plt.close()

            plot_name = 'cond_im_grid_' + tag
            name_gif = f'{iter_name}_{plot_name}'
            filename_grid_png = os.path.join(im_dir, plot_name + '_last.png')
            save_image_with_metrics(x_tot_cond[:, -1], filename_grid_png)
            self.ipf.save_logger.log_image(f"cond/{plot_name}_last", [filename_grid_png], step=self.step, fb=fb)

            # if self.plot_level >= 2:
            #     plot_paths = []
            #     x_start_tot = torch.cat([x_start.unsqueeze(1), x_tot_cond], dim=1)
            #     for k in range(self.num_steps+1):
            #         if k % freq == 0 or k == self.num_steps:
            #             # save png
            #             filename_grid_png = os.path.join(im_dir, f'{plot_name}_{k}.png')
            #             plot_paths.append(filename_grid_png)
            #             save_image_with_metrics(x_start_tot[:, k], filename_grid_png)
            #
            #     make_gif(plot_paths, output_directory=gif_dir, gif_name=name_gif)

            if self.plot_level >= 3:
                im_dir = os.path.join(im_dir, "im_" + tag)
                os.makedirs(im_dir, exist_ok=True)
                for j in range(len(y_cond)):
                    im_dir_j = os.path.join(im_dir, str(j))
                    os.mkdir(im_dir_j)
                    for k in range(x_tot_cond.shape[2]):
                        plt.clf()
                        filename_png = os.path.join(im_dir_j, '{:05}.png'.format(k))
                        save_image(x_tot_cond[j, -1, k], filename_png)

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        self.plot_sequence_cond(x_tot_fwd[:, -1], y_cond, x_tot_cond, data, i, n, fb, x_init_cond=x_init_cond, tag=tag, freq=freq)

    def plot_and_record_batch_joint(self, x_last, x_init, iters, i, n, fb, dl_name='train'):
        if fb == 'b':
            uint8_x_init = to_uint8_tensor(x_init)
            uint8_x_last = to_uint8_tensor(x_last)

            for metric in self.metrics_dict.values():
                metric.update(uint8_x_last, uint8_x_init)

            if self.plot_level >= 3:
                name = str(i) + '_' + fb + '_' + str(n)
                im_dir = os.path.join(self.im_dir, name, dl_name)
                im_dir = os.path.join(im_dir, "im/")
                os.makedirs(im_dir, exist_ok=True)

                for k in range(x_last.shape[0]):
                    plt.clf()
                    file_idx = iters * self.ipf.test_batch_size + self.ipf.accelerator.process_index * self.ipf.test_batch_size // self.ipf.accelerator.num_processes + k
                    filename_png = os.path.join(im_dir, '{:05}.png'.format(file_idx))
                    assert not os.path.isfile(filename_png)
                    save_image(x_last[k], filename_png)


class OneDCondPlotter(Plotter):
    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', freq=None,
                            mean_final=None, var_final=None):
        if freq is None:
            freq = self.num_steps//min(self.num_steps,50)
        iter_name = str(i) + '_' + fb + '_' + str(n)
        im_dir = os.path.join(self.im_dir, iter_name, dl_name)
        gif_dir = os.path.join(self.gif_dir, dl_name)

        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        ylim = [-3, 3]
        npts = 250
        if data == 'type1':
            xlim = [-1,3]
            # true_pdf = lambda xy: 1/6 * gamma.pdf(xy[:, 0] - np.tanh(xy[:, 1]), 1, scale=0.3)
        elif data == 'type2':
            xlim = [-1,1]
            # def true_pdf(xy):
            #     x_pyt = torch.from_numpy(xy[:, 0])
            #     y_pyt = torch.from_numpy(xy[:, 1])
            #     base_distribution = torch.distributions.Normal(y_pyt, np.sqrt(0.05))
            #     transforms = [torch.distributions.transforms.TanhTransform()]
            #     return 1/6 * torch.distributions.TransformedDistribution(
            #         base_distribution, transforms).log_prob(x_pyt).exp().numpy()
        elif data == 'type3':
            xlim = [-0.8,0.8]
            # true_pdf = lambda xy: 1/6 * gamma.pdf(xy[:, 0] / np.tanh(xy[:, 1]), 1, scale=0.3)

        # DENSITY
        # ROLES OF X AND Y inversed when compared to Conditional Sampling.

        x_start = x_start.cpu().numpy()
        y_start = y_start.cpu().numpy()
        x_tot = x_tot.cpu().numpy()

        if n == 0 and fb == "f":
            plt.clf()
            filename = f'true_density_{dl_name}.png'
            filename = os.path.join(self.im_dir, filename)
            kde_yx = kde.gaussian_kde([y_start[:,0], x_start[:,0]])
            xi, yi = np.mgrid[ylim[0]:ylim[1]:npts*1j, xlim[0]:xlim[1]:npts*1j]
            zi = kde_yx(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
            plt.xlabel("y")
            plt.ylabel("x")
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)

            self.ipf.save_logger.log_image(dl_name + '/true_density', [filename], step=self.step, fb=fb)

        plot_name = 'density'
        name_gif = f'{iter_name}_{plot_name}'
        plot_paths_reg = []
        x_start_tot = np.concatenate([np.expand_dims(x_start, axis=0), x_tot], axis=0)
        for k in range(self.num_steps+1):
            if k % freq == 0 or k == self.num_steps:
                filename = plot_name + '_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)
                plt.clf()
                if n is not None:
                    str_title = 'IPFP iteration: ' + str(n)
                    plt.title(str_title)
                kde_yx = kde.gaussian_kde([y_start[:, 0], x_start_tot[k, :, 0]])
                xi, yi = np.mgrid[ylim[0]:ylim[1]:npts*1j, xlim[0]:xlim[1]:npts*1j]
                zi = kde_yx(np.vstack([xi.flatten(), yi.flatten()]))
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                plt.xlabel("y")
                plt.ylabel("x")
                plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

        self.ipf.save_logger.log_image(f'{dl_name}/{plot_name}_last', [filename], step=self.step, fb=fb)

    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        if freq is None:
            freq = self.num_steps//min(self.num_steps,50)
        iter_name = str(i) + '_' + fb + '_' + str(n) + '_'
        im_dir = os.path.join(self.im_dir, iter_name, 'cond')
        gif_dir = os.path.join(self.gif_dir, 'cond')

        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        npts = 250
        if data == 'type1':
            xlim = [-1,3]
            colors = ['red', 'green', 'blue']
        elif data == 'type2':
            xlim = [-1,1]
            colors = ['red', 'green', 'blue']
        elif data == 'type3':
            xlim = [-0.8,0.8]
            colors = ['red', 'blue']

        x_start = x_start.cpu().numpy()
        x_tot_cond = x_tot_cond.cpu().numpy()

        # HISTOGRAMS
        plot_name = 'cond_histogram_' + tag
        name_gif = f'{iter_name}_{plot_name}'
        plot_paths_reg = []

        x_lin = np.linspace(xlim[0], xlim[1], npts)
        zs_lin = np.zeros([0, npts])

        y_cond = y_cond.cpu().numpy()

        for j in range(len(y_cond)):
            y_c = y_cond[j]

            if data == 'type1':
                z = gamma.pdf(x_lin - np.tanh(y_c), 1, scale=0.3)
            elif data == 'type2':
                sigma = np.sqrt(0.05)
                z1 = 1 / (1 - x_lin**2)
                z2 = np.arctanh(x_lin)
                z3 = norm.pdf(z2, loc=y_c, scale=sigma)
                z = z3 * z1
            elif data == 'type3':
                z = gamma.pdf(x_lin / np.tanh(y_c), 1, scale=0.3)

            zs_lin = np.vstack([zs_lin, z])

        x_start_tot_cond = np.concatenate([np.expand_dims(x_start, axis=1), x_tot_cond], axis=1)
        for k in range(self.num_steps+1):
            if k % freq == 0 or k == self.num_steps:
                plt.clf()
                for j in range(len(y_cond)):
                    y_c = y_cond[j]

                    x_cond = x_start_tot_cond[j][k, :, 0]

                    plt.plot(x_lin, zs_lin[j], color=colors[j])
                    plt.hist(x_cond, bins=50, range=(xlim[0], xlim[1]), density=True, color=colors[j])

                filename = plot_name + '_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)

                if n is not None:
                    str_title = 'IPFP iteration: ' + str(n)
                    plt.title(str_title)
                plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

        self.ipf.save_logger.log_image(f'cond/{plot_name}_last', [filename], step=self.step, fb=fb)

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        self.plot_sequence_cond(x_tot_fwd[:, -1], y_cond, x_tot_cond, data, i, n, fb, x_init_cond=x_init_cond, tag=tag, freq=freq)

    def test_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', mean_final=None, var_final=None):
        out = super().test_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, mean_final=mean_final, var_final=var_final)

        if fb == 'b':
            y_start = y_start.detach().cpu().numpy()
            x_last = x_tot[-1].detach().cpu().numpy()
            last_kde = lambda xy: kde.gaussian_kde([x_last[:, 0], y_start[:, 0]])(xy.T)

            x_init = x_init.detach().cpu().numpy()
            data_kde = lambda xy: kde.gaussian_kde([x_init[:, 0], y_start[:, 0]])(xy.T)

            batch = np.hstack([x_init, y_start])

            out[dl_name + "/l2_pq"] = np.mean((data_kde(batch) - last_kde(batch)) ** 2)
            out[dl_name + "/kl_pq"] = np.mean(np.log(data_kde(batch)) - np.log(last_kde(batch)))

        return out


class OneDRevCondPlotter(Plotter):
    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', freq=None,
                            mean_final=None, var_final=None):
        if freq is None:
            freq = self.num_steps // min(self.num_steps, 50)
        iter_name = str(i) + '_' + fb + '_' + str(n)
        im_dir = os.path.join(self.im_dir, iter_name, dl_name)
        gif_dir = os.path.join(self.gif_dir, dl_name)

        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        npts = 250
        xlim = [-3*self.args.data.x_std, 3*self.args.data.x_std]
        if data == 'type1':
            ylim = [-1, 6]
        elif data == 'type4':
            ylim = [-1 - 3*(self.args.data.x_std+self.args.data.y_std), 1 + 3*(self.args.data.x_std+self.args.data.y_std)]

        # DENSITY

        x_start = x_start.cpu().numpy()
        y_start = y_start.cpu().numpy()
        x_tot = x_tot.cpu().numpy()

        if n == 0 and fb == "f":
            plt.clf()
            filename = f'true_density_{dl_name}.png'
            filename = os.path.join(self.im_dir, filename)
            g = sns.kdeplot(x=x_start[:, 0], y=y_start[:, 0])
            g.set(xlim=xlim)
            g.set(ylim=ylim)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)

        plot_name = 'density'
        name_gif = f'{iter_name}_{plot_name}'
        plot_paths_reg = []
        x_start_tot = np.concatenate([np.expand_dims(x_start, axis=0), x_tot], axis=0)
        for k in range(self.num_steps+1):
            if k % freq == 0 or k == self.num_steps:
                filename = plot_name + '_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)
                plt.clf()
                if n is not None:
                    str_title = 'IPFP iteration: ' + str(n)
                    plt.title(str_title)
                g = sns.kdeplot(x=x_start_tot[k, :, 0], y=y_start[:, 0])
                g.set(xlim=xlim)
                g.set(ylim=ylim)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)


class FiveDCondPlotter(Plotter):
    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        if freq is None:
            freq = self.num_steps//min(self.num_steps,50)
        iter_name = str(i) + '_' + fb + '_' + str(n) + '_'
        im_dir = os.path.join(self.im_dir, iter_name, 'cond')
        gif_dir = os.path.join(self.gif_dir, 'cond')

        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        npts = 250
        if data == 'type1':
            xlim = [-4.5,6.5]
        elif data == 'type2':
            xlim = [-4,9]
        elif data == 'type3':
            xlim = [-5,125]
        elif data == 'type4':
            xlim = [-3.5,3.5]

        x_start = x_start.cpu().numpy()
        x_tot_cond = x_tot_cond.cpu().numpy()

        # HISTOGRAMS
        plot_name = 'cond_histogram_' + tag
        name_gif = f'{iter_name}_{plot_name}'
        plot_paths_reg = []

        if fb == 'b' and y_cond is not None:
            x_lin = np.linspace(xlim[0], xlim[1], npts)
            zs_lin = np.zeros([0, npts])

            y_cond = y_cond.cpu().numpy()

            for j in range(len(y_cond)):
                y_c = y_cond[j]

                if data == 'type1':
                    z = norm.pdf(x_lin, loc=y_c[0]**2 + torch.exp(y_c[1] + y_c[2]/3) + torch.sin(y_c[3] + y_c[4]))

                elif data == 'type2':
                    x_mean = y_c[0]**2 + torch.exp(y_c[1] + y_c[2]/3) + y_c[3] - y_c[4]
                    x_std = 0.5 + y_c[1]**2/2 + y_c[4]**2/2
                    z = norm.pdf(x_lin, loc=x_mean, scale=x_std)

                elif data == 'type3':
                    mult = 5 + y_c[0]**2/3 + y_c[1]**2 + y_c[2]**2 + y_c[3] + y_c[4]
                    z = 0.5 * lognorm.pdf(x_lin, s=0.5, scale=np.exp(1)*mult) + 0.5 * lognorm.pdf(x_lin, s=0.5, scale=np.exp(-1)*mult)

                elif data == 'type4':
                    z = 0.5 * norm.pdf(x_lin, loc=-y_c[0], scale=0.25) + 0.5 * norm.pdf(x_lin, loc=y_c[0], scale=0.25)

                zs_lin = np.vstack([zs_lin, z])

            x_start_tot_cond = np.concatenate([np.expand_dims(x_start, axis=1), x_tot_cond], axis=1)
            for k in range(self.num_steps+1):
                if k % freq == 0 or k == self.num_steps:
                    plt.clf()
                    for j in range(len(y_cond)):
                        y_c = y_cond[j]

                        x_cond = x_start_tot_cond[j][k, :, 0]

                        x_cond_kde = kde.gaussian_kde(x_cond)(x_lin)

                        plt.plot(x_lin, zs_lin[j], color="C"+str(j))
                        plt.plot(x_lin, x_cond_kde, color="C"+str(j), ls="--")

                    filename = plot_name + '_' + str(k) + '.png'
                    filename = os.path.join(im_dir, filename)

                    if n is not None:
                        str_title = 'IPFP iteration: ' + str(n)
                        plt.title(str_title)
                    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                    plot_paths_reg.append(filename)

            make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)

            raw_data_save = {'x': x_lin, 'y': y_cond, 'px_y_true': zs_lin}
            zs_cond_kde = np.zeros([0, npts])
            for j in range(len(y_cond)):
                x_cond = x_start_tot_cond[j][k, :, 0]

                z_cond_kde = kde.gaussian_kde(x_cond)(x_lin)
                zs_cond_kde = np.vstack([zs_cond_kde, z_cond_kde])
            raw_data_save['px_y_kde'] = zs_cond_kde
            torch.save(raw_data_save, os.path.join(im_dir, plot_name + '_raw_data_' + str(k) + '.pt'))

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        self.plot_sequence_cond(x_tot_fwd[:, -1], y_cond, x_tot_cond, data, i, n, fb, x_init_cond=x_init_cond, tag=tag, freq=freq)

    def test_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag=''):
        out = super().test_cond(x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=x_init_cond, tag=tag)

        if fb == 'b' and y_cond is not None:
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

            x_tot_cond_std, x_tot_cond_mean = torch.std_mean(x_tot_cond[:, -1], 1)

            out["cond/mse_mean_" + tag] = torch.mean((x_tot_cond_mean - true_x_test_mean)**2)
            out["cond/mse_std_" + tag] = torch.mean((x_tot_cond_std - true_x_test_std)**2)

        return out


class BiochemicalPlotter(Plotter):
    def plot(self, x_start, x_tot, y_tot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward):
        fb = forward_or_backward
        ipf_it = n
        x_tot = x_tot.cpu().numpy()
        y_tot = y_tot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        self.save_sequence_cond_biochemical(num_steps=self.num_steps, x=x_tot, y=y_tot,
                           data=data, init_dl=init_dl, y_cond=y_cond,
                           x_tot_cond=x_tot_cond, fb=fb, name=name,
                           ipf_it=ipf_it,
                           freq=self.num_steps//min(self.num_steps,50),
                           im_dir=self.im_dir, gif_dir=self.gif_dir)

    @staticmethod
    def save_sequence_cond_biochemical(num_steps, x, y, data, init_dl, y_cond, x_tot_cond, fb, name='', im_dir='./im', gif_dir = './gif', ipf_it=None, freq=1):
        xlim = [-0.5, 1.5]
        ylim = [-0.5, 2.5]
        npts = 100

        x_tot = x_tot_cond[0].cpu().numpy()

        # DENSITY
        # ROLES OF X AND Y inversed when compared to Conditional Sampling.

        name_gif = name + 'density'
        plot_paths_reg = []
        for k in range(num_steps):
            if k % freq == 0 or k == num_steps:
                filename =  name + 'density_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)
                plt.clf()            
                if ipf_it is not None:
                    str_title = 'IPFP iteration: ' + str(ipf_it)
                    plt.title(str_title)
                kde_xy = kde.gaussian_kde([x_tot[k, :, 0],x_tot[k, :, 1]])
                xi, yi = np.mgrid[xlim[0]:xlim[1]:npts*1j, ylim[0]:ylim[1]:npts*1j]
                zi = kde_xy(np.vstack([xi.flatten(), yi.flatten()]))
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)    
    

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.utils as vutils
from ..data.utils import save_image
from PIL import Image
import os, sys
matplotlib.use('Agg')
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

    def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif'):
        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

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
        os.makedirs(self.im_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)

        self.num_steps = num_steps
        self.gammas = gammas

    def __call__(self, x_start, y_start, x_tot, y_cond, x_tot_cond, x_init, data, i, n, fb, x_init_cond=None,
                 mean_final=None, var_final=None):
        self.plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb,
                                 mean_final=mean_final, var_final=var_final)
        self.plot_sequence_cond(x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=x_init_cond)

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, tag='', freq=None,
                            mean_final=None, var_final=None):
        if freq is None:
            freq = self.num_steps // min(self.num_steps, 50)
        name = str(i) + '_' + fb + '_' + str(n) + '_'
        im_dir = os.path.join(self.im_dir, name)
        name = name + tag

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)

        x_tot = x_tot.cpu().reshape(x_tot.shape[0], x_tot.shape[1], -1).numpy()

        name_gif = name + '_histogram'
        plot_paths_reg = []
        dims = np.sort(np.random.choice(x_tot.shape[-1], min(x_tot.shape[-1], 3), replace=False))
        for k in range(self.num_steps):
            if k % freq == 0:
                filename = name_gif + '_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)
                plt.clf()
                fig = plt.figure(figsize=(5*len(dims), 4))
                if n is not None:
                    str_title = 'IPFP iteration: ' + str(n)
                    plt.title(str_title)

                for d in range(len(dims)):
                    dim = dims[d]
                    x_min, x_max = np.min(x_tot[:, :, dim]), np.max(x_tot[:, :, dim])
                    ax = plt.subplot(1, len(dims), d+1)
                    plt.hist(x_tot[k, :, dim], bins=50, density=True, alpha=0.5)
                    if mean_final is not None and var_final is not None and fb == 'f':
                        if (mean_final + x_start[:1]).shape[0] == 1 and (var_final + x_start[:1]).shape[0] == 1:
                            mu = mean_final.view(-1)[dim]
                            sig = np.sqrt(var_final.view(-1)[dim])
                            x_lin = np.linspace(x_min, x_max, 250)
                            plt.plot(x_lin, stats.norm.pdf(x_lin, mu, sig))
                    ax.set_xlim(x_min, x_max)
                plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                plt.close()
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)

        if x_start[0].shape == y_start[0].shape:
            npts = 250
            x_start = x_start.cpu().reshape(x_start.shape[0], -1).numpy()
            y_start = y_start.cpu().reshape(y_start.shape[0], -1).numpy()

            if n == 0 and fb == "f":
                plt.clf()
                filename = 'original_density.png'
                filename = os.path.join(self.im_dir, filename)

                fig = plt.figure(figsize=(5*len(dims), 4))
                for d in range(len(dims)):
                    dim = dims[d]
                    x_min, x_max = np.min(x_tot[:, :, dim]), np.max(x_tot[:, :, dim])
                    y_min, y_max = np.min(y_start[:, dim]), np.max(y_start[:, dim])
                    ax = plt.subplot(1, len(dims), d+1)
                    kde_yx = kde.gaussian_kde([y_start[:, dim], x_start[:, dim]])
                    xi, yi = np.mgrid[y_min:y_max:npts * 1j, x_min:x_max:npts * 1j]
                    zi = kde_yx(np.vstack([xi.flatten(), yi.flatten()]))
                    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                    plt.xlabel("y")
                    plt.ylabel("x")
                plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                plt.close()

            name_gif = name + '_density'
            plot_paths_reg = []
            for k in range(self.num_steps):
                if k % freq == 0:
                    filename = name_gif + '_' + str(k) + '.png'
                    filename = os.path.join(im_dir, filename)
                    plt.clf()
                    fig = plt.figure(figsize=(5*len(dims), 4))
                    if n is not None:
                        str_title = 'IPFP iteration: ' + str(n)
                        plt.title(str_title)

                    for d in range(len(dims)):
                        dim = dims[d]
                        x_min, x_max = np.min(x_tot[:, :, dim]), np.max(x_tot[:, :, dim])
                        y_min, y_max = np.min(y_start[:, dim]), np.max(y_start[:, dim])
                        ax = plt.subplot(1, len(dims), d+1)
                        kde_yx = kde.gaussian_kde([y_start[:, dim], x_tot[k, :, dim]])
                        xi, yi = np.mgrid[y_min:y_max:npts * 1j, x_min:x_max:npts * 1j]
                        zi = kde_yx(np.vstack([xi.flatten(), yi.flatten()]))
                        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                        plt.xlabel("y")
                        plt.ylabel("x")
                    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=DPI)
                    plt.close()
                    plot_paths_reg.append(filename)

            make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)

    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        pass

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        pass


class ImPlotter(Plotter):

    def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif', plot_level=3):
        super().__init__(num_steps, gammas, im_dir, gif_dir)
        self.num_plots = 100
        # self.num_digits = 20
        self.plot_level = plot_level

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, tag='', freq=None,
                            mean_final=None, var_final=None):
        super().plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, tag=tag, freq=freq,
                                    mean_final=mean_final, var_final=var_final)
        if freq is None:
            freq = self.num_steps // min(self.num_steps, 50)

        if self.plot_level > 0:
            x_tot = x_tot[:, :self.num_plots]
            name = str(i) + '_' + fb + '_' + str(n) + '_'
            im_dir = os.path.join(self.im_dir, name)
            name = name + tag

            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)

            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(im_dir, 'im_grid_start.png')
                save_image(x_start, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_last.png')
                save_image(x_tot[-1], filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_data.png')
                save_image(x_init, filename_grid_png, nrow=10)

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []

                for k in range(self.num_steps):
                    if k % freq == 0:
                        # save png
                        filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))
                        plot_paths.append(filename_grid_png)
                        save_image(x_tot[k], filename_grid_png, nrow=10)

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name+'_samples')


# class TwoDPlotter(Plotter):
#
#     def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif'):
#
#         if not os.path.isdir(im_dir):
#             os.mkdir(im_dir)
#         if not os.path.isdir(gif_dir):
#             os.mkdir(gif_dir)
#
#         self.im_dir = im_dir
#         self.gif_dir = gif_dir
#
#         self.num_steps = num_steps
#         self.gammas = gammas
#
#     def plot(self, x_start, x_tot, i, n, forward_or_backward):
#         fb = forward_or_backward
#         ipf_it = n
#         x_tot = x_tot.cpu().numpy()
#         name = str(i) + '_' + fb +'_' + str(n) + '_'
#
#         save_sequence(num_steps=self.num_steps, x=x_tot, name=name,
#                       xlim=(-15,15), ylim=(-15,15), ipf_it=ipf_it,
#                       freq=self.num_steps//min(self.num_steps,50),
#                       im_dir=self.im_dir, gif_dir=self.gif_dir)
#
#
#     def __call__(self, x_start, x_tot, i, n, forward_or_backward):
#         self.plot(x_start, x_tot, i, n, forward_or_backward)


class OneDCondPlotter(Plotter):
    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, tag='', freq=None,
                            mean_final=None, var_final=None):
        if freq is None:
            freq = self.num_steps//min(self.num_steps,50)
        name = str(i) + '_' + fb + '_' + str(n) + '_'
        im_dir = os.path.join(self.im_dir, name)
        name = name + tag

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)

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
            filename = 'original_density.png'
            filename = os.path.join(self.im_dir, filename)
            kde_yx = kde.gaussian_kde([y_start[:,0], x_start[:,0]])
            xi, yi = np.mgrid[ylim[0]:ylim[1]:npts*1j, xlim[0]:xlim[1]:npts*1j]
            zi = kde_yx(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
            plt.xlabel("y")
            plt.ylabel("x")
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)

        name_gif = name + '_density'
        plot_paths_reg = []
        for k in range(self.num_steps):
            if k % freq == 0:
                filename = name_gif + '_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)
                plt.clf()            
                if n is not None:
                    str_title = 'IPFP iteration: ' + str(n)
                    plt.title(str_title)
                kde_yx = kde.gaussian_kde([y_start[:, 0], x_tot[k, :, 0]])
                xi, yi = np.mgrid[ylim[0]:ylim[1]:npts*1j, xlim[0]:xlim[1]:npts*1j]
                zi = kde_yx(np.vstack([xi.flatten(), yi.flatten()]))
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                plt.xlabel("y")
                plt.ylabel("x")
                plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)    

    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        if freq is None:
            freq = self.num_steps//min(self.num_steps,50)
        name = str(i) + '_' + fb + '_' + str(n) + '_'
        im_dir = os.path.join(self.im_dir, name)
        name = name + tag

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        
        ylim = [-3, 3]
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
        
        # HISTOGRAMS
        name_gif = name + '_cond_histogram'
        plot_paths_reg = []

        if fb == 'b' and y_cond is not None:
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

            for k in range(self.num_steps):
                if k % freq == 0:
                    plt.clf()
                    for j in range(len(y_cond)):
                        y_c = y_cond[j]

                        x_cond = x_tot_cond[j][k, :, 0].cpu().numpy()

                        plt.plot(x_lin, zs_lin[j], color=colors[j])
                        plt.hist(x_cond, bins=50, range=(xlim[0], xlim[1]), density=True, color=colors[j])

                    filename = name_gif + '_' + str(k) + '.png'
                    filename = os.path.join(im_dir, filename)

                    if n is not None:
                        str_title = 'IPFP iteration: ' + str(n)
                        plt.title(str_title)
                    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                    plot_paths_reg.append(filename)
        
            make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        self.plot_sequence_cond(x_tot_fwd[-1], y_cond, x_tot_cond, data, i, n, fb, tag=tag, freq=freq)


class FiveDCondPlotter(Plotter):
    def plot_sequence_cond(self, x_start, y_cond, x_tot_cond, data, i, n, fb, x_init_cond=None, tag='', freq=None):
        if freq is None:
            freq = self.num_steps//min(self.num_steps,50)
        name = str(i) + '_' + fb + '_' + str(n) + '_'
        im_dir = os.path.join(self.im_dir, name)
        name = name + tag

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        
        npts = 250
        if data == 'type1':
            xlim = [-4.5,6.5]
        elif data == 'type2':
            xlim = [-4,9]
        elif data == 'type3':
            xlim = [-5,125]
        elif data == 'type4':
            xlim = [-3.5,3.5]

        # HISTOGRAMS
        name_gif = name + '_cond_histogram'
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

            for k in range(self.num_steps):
                if k % freq == 0:
                    plt.clf()
                    for j in range(len(y_cond)):
                        y_c = y_cond[j]

                        x_cond = x_tot_cond[j][k, :, 0].cpu().numpy()

                        x_cond_kde = kde.gaussian_kde(x_cond)(x_lin)

                        plt.plot(x_lin, zs_lin[j], color="C"+str(j))
                        plt.plot(x_lin, x_cond_kde, color="C"+str(j), ls="--")

                    filename = name_gif + '_' + str(k) + '.png'
                    filename = os.path.join(im_dir, filename)

                    if n is not None:
                        str_title = 'IPFP iteration: ' + str(n)
                        plt.title(str_title)
                    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                    plot_paths_reg.append(filename)
                
                if k == self.num_steps - 1:
                    raw_data_save = {'x': x_lin, 'y': y_cond, 'px_y_true': zs_lin}
                    zs_cond_kde = np.zeros([0, npts])
                    for j in range(len(y_cond)):
                        y_c = y_cond[j]

                        x_cond = x_tot_cond[j][k, :, 0].cpu().numpy()

                        z_cond_kde = kde.gaussian_kde(x_cond)(x_lin)
                        zs_cond_kde = np.vstack([zs_cond_kde, z_cond_kde])
                    raw_data_save['px_y_kde'] = zs_cond_kde
                    torch.save(raw_data_save, os.path.join(im_dir, name_gif + '_raw_data_' + str(k) + '.pt'))

            make_gif(plot_paths_reg, output_directory=self.gif_dir, gif_name=name_gif)

    def plot_sequence_cond_fwdbwd(self, x_init, y_init, x_tot_fwd, y_cond, x_tot_cond, data, i, n, fb,
                                  x_init_cond=None, tag='fwdbwd', freq=None):
        self.plot_sequence_cond(x_tot_fwd[-1], y_cond, x_tot_cond, data, i, n, fb, tag=tag, freq=freq)


class BiochemicalPlotter(Plotter):

    def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif'):

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.num_steps = num_steps
        self.gammas = gammas

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
            if k % freq == 0:
                filename =  name + 'density_' + str(k) + '.png'
                filename = os.path.join(im_dir, filename)
                plt.clf()            
                if ipf_it is not None:
                    str_title = 'IPFP iteration: ' + str(ipf_it)
                    plt.title(str_title)
                k = kde.gaussian_kde([x_tot[k, :, 0],x_tot[k, :, 1]])
                xi, yi = np.mgrid[xlim[0]:xlim[1]:npts*1j, ylim[0]:ylim[1]:npts*1j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
                plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                plot_paths_reg.append(filename)

        make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)    
    

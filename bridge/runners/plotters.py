import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision.utils as vutils
from PIL import Image
from ..data.two_dim import data_distrib
import os, sys
matplotlib.use('Agg')
from scipy.stats import kde, gamma



DPI = 200

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)
    
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
    

def save_sequence_cond_1d(num_steps, x, y, data, init_dl, y_cond, x_tot_cond, fb, name='', im_dir='./im', gif_dir = './gif', ipf_it=None, freq=1):

    ylim = [-3, 3]
    npts = 100
    if data == 'type1':
        xlim = [-1,3]
        y_cond = [-1.2, 0, 1.2]
        colors = ['green', 'orange', 'blue']
    elif data == 'type2':
        xlim = [-.95,.95]
        y_cond = [-1.2, 0, 1.2]
        colors = ['green', 'orange', 'blue']
    else:
        xlim = [-0.8,0.8]
        y_cond = [-1.2, 1.2]
        colors = ['green', 'orange']
    
    # DENSITY
    # ROLES OF X AND Y inversed when compared to Conditional Sampling.
    
    val_x = torch.zeros(0,1)
    val_y = torch.zeros(0,1)
    
    if ipf_it == 0:
        filename = 'original_density.png'
        filename = os.path.join(im_dir, filename)
        N_APPROX = 10
        for n in range(N_APPROX):
            batch = next(init_dl)
            batch_x = batch[0]
            batch_y = batch[1]
            val_x = torch.cat((val_x, batch_x), 0)
            val_y = torch.cat((val_y, batch_y), 0)
        val_x = val_x.cpu().numpy()
        val_y = val_y.cpu().numpy()
        k = kde.gaussian_kde([val_y[:,0],val_x[:,0]])
        xi, yi = np.mgrid[ylim[0]:ylim[1]:npts*1j, xlim[0]:xlim[1]:npts*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)                

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
            k = kde.gaussian_kde([y[k, :, 0],x[k, :, 0]])
            xi, yi = np.mgrid[ylim[0]:ylim[1]:npts*1j, xlim[0]:xlim[1]:npts*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
            plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
            plot_paths_reg.append(filename)

    make_gif(plot_paths_reg, output_directory=gif_dir, gif_name=name_gif)    

    # HISTOGRAMS
    name_gif = name + 'histogram'
    plot_paths_reg = []
    abc = torch.zeros(0, 1)

    if fb == 'b':
        for n in range(len(y_cond)):
            y_c = y_cond[n]
            x_lin = np.linspace(xlim[0],xlim[1],npts)

            if data == 'type1':
                z = gamma.pdf(x_lin, 0.3, loc=np.tanh(y_c), scale=1)
            elif data == 'type2':
                sigma = np.sqrt(0.05)
                z1 = (1 - x_lin ** 2)**(-1)/sigma
                z2 = (np.arctanh(x_lin) - y_c)/sigma
                z3 = np.sqrt(2 * np.pi) ** (-1) * np.exp(-z2**2)
                z = z3 * z1

            for k in range(num_steps):
                if k % freq == 0:
                    filename =  name + 'histogram_' + str(n) + '_' + str(k) + '.png'
                    filename = os.path.join(im_dir, filename)
                    plt.clf()            
                    if ipf_it is not None:
                        str_title = 'IPFP iteration: ' + str(ipf_it)
                        plt.title(str_title)

                    x_cond = x_tot_cond[n][k, :, 0]
                    x_cond_np = x_cond.cpu().numpy()
                    plt.clf()
                    plt.plot(x_lin, z)
                    plt.hist(x_cond_np, bins=30, range=(xlim[0], xlim[1]), density=True)
                    plt.savefig(filename, bbox_inches = 'tight', transparent = True, dpi=DPI)
                    plot_paths_reg.append(filename)
    
    
class Plotter(object):

    def __init__(self):
        pass

    def plot(self, x_tot_plot, net, i, n, forward_or_backward):
        pass

    def __call__(self, initial_sample, x_tot_plot, net, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, net, i, n, forward_or_backward)



class ImPlotter(object):

    def __init__(self, im_dir = './im', gif_dir='./gif', plot_level=3):
        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)
        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.num_plots = 100
        self.num_digits = 20
        self.plot_level = plot_level
        

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        if self.plot_level > 0:
            x_tot_plot = x_tot_plot[:,:self.num_plots]
            name = '{0}_{1}_{2}'.format(forward_or_backward, n, i)
            im_dir = os.path.join(self.im_dir, name)
            
            if not os.path.isdir(im_dir):
                os.mkdir(im_dir)         
            
            if self.plot_level > 0:
                plt.clf()
                filename_grid_png = os.path.join(im_dir, 'im_grid_first.png')
                vutils.save_image(initial_sample, filename_grid_png, nrow=10)
                filename_grid_png = os.path.join(im_dir, 'im_grid_final.png')
                vutils.save_image(x_tot_plot[-1], filename_grid_png, nrow=10)

            if self.plot_level >= 2:
                plt.clf()
                plot_paths = []
                num_steps, num_particles, channels, H, W = x_tot_plot.shape
                plot_steps = np.linspace(0,num_steps-1,self.num_plots, dtype=int) 

                for k in plot_steps:
                    # save png
                    filename_grid_png = os.path.join(im_dir, 'im_grid_{0}.png'.format(k))    
                    plot_paths.append(filename_grid_png)
                    vutils.save_image(x_tot_plot[k], filename_grid_png, nrow=10)
                    

                make_gif(plot_paths, output_directory=self.gif_dir, gif_name=name)

    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


class TwoDPlotter(Plotter):

    def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif'):

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.num_steps = num_steps
        self.gammas = gammas

    def plot(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        fb = forward_or_backward
        ipf_it = n
        x_tot_plot = x_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        save_sequence(num_steps=self.num_steps, x=x_tot_plot, name=name,
                      xlim=(-15,15), ylim=(-15,15), ipf_it=ipf_it,
                      freq=self.num_steps//min(self.num_steps,50),
                      im_dir=self.im_dir, gif_dir=self.gif_dir)


    def __call__(self, initial_sample, x_tot_plot, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, i, n, forward_or_backward)


class OneDCondPlotter(Plotter):

    def __init__(self, num_steps, gammas, im_dir = './im', gif_dir='./gif'):

        if not os.path.isdir(im_dir):
            os.mkdir(im_dir)
        if not os.path.isdir(gif_dir):
            os.mkdir(gif_dir)

        self.im_dir = im_dir
        self.gif_dir = gif_dir

        self.num_steps = num_steps
        self.gammas = gammas

    def plot(self, initial_sample, x_tot_plot, y_tot_plot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward):
        fb = forward_or_backward
        ipf_it = n
        x_tot_plot = x_tot_plot.cpu().numpy()
        y_tot_plot = y_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        save_sequence_cond_1d(num_steps=self.num_steps, x=x_tot_plot, y=y_tot_plot,
                           data=data, init_dl=init_dl, y_cond=y_cond,
                           x_tot_cond=x_tot_cond, fb=fb, name=name,
                           ipf_it=ipf_it,
                           freq=self.num_steps//min(self.num_steps,50),
                           im_dir=self.im_dir, gif_dir=self.gif_dir)


    def __call__(self, initial_sample, x_tot_plot, y_tot_plot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, y_tot_plot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward)


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

    def plot(self, initial_sample, x_tot_plot, y_tot_plot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward):
        fb = forward_or_backward
        ipf_it = n
        x_tot_plot = x_tot_plot.cpu().numpy()
        y_tot_plot = y_tot_plot.cpu().numpy()
        name = str(i) + '_' + fb +'_' + str(n) + '_'

        save_sequence_cond_biochemical(num_steps=self.num_steps, x=x_tot_plot, y=y_tot_plot,
                           data=data, init_dl=init_dl, y_cond=y_cond,
                           x_tot_cond=x_tot_cond, fb=fb, name=name,
                           ipf_it=ipf_it,
                           freq=self.num_steps//min(self.num_steps,50),
                           im_dir=self.im_dir, gif_dir=self.gif_dir)


    def __call__(self, initial_sample, x_tot_plot, y_tot_plot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward):
        self.plot(initial_sample, x_tot_plot, y_tot_plot, data, init_dl, y_cond, x_tot_cond, i, n, forward_or_backward)



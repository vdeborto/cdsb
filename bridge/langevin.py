import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np 

def grad_gauss(x, m, var):
    xout = (x - m) / var
    return -xout

def ornstein_ulhenbeck(x, gradx, gamma):
    xout = x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
    return xout

class Langevin(torch.nn.Module):

    def __init__(self, num_steps, shape_x, shape_y, gammas, time_sampler, device = None, 
                 mean_final=torch.tensor([0.,0.]), var_final=torch.tensor([.5, .5]), mean_match=True):
        super().__init__()

        self.mean_match = mean_match
        self.mean_final = mean_final
        self.var_final = var_final
        
        self.num_steps = num_steps # num diffusion steps
        self.d_x = shape_x # dimension of object to diffuse
        self.d_y = shape_y # dimension of conditioning
        self.gammas = gammas.float() # schedule
        gammas_vec = torch.ones(self.num_steps,*self.d_x,device=device)
        for k in range(num_steps):
            gammas_vec[k] = gammas[k].float()
        self.gammas_vec = gammas_vec    

        if device is not None:
            self.device = device
        else:
            self.device = gammas.device

        self.steps = torch.arange(self.num_steps).to(self.device)
        self.time = torch.cumsum(self.gammas,0).to(self.device).float()
        self.time_sampler = time_sampler
            

    def record_init_langevin(self, init_samples_x, init_samples_y, mean_final=None, var_final=None):
        if mean_final is None:
            mean_final = self.mean_final
        if var_final is None:
            var_final = self.var_final
        
        x = init_samples_x
        y = init_samples_y
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))


        x_tot = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        y_tot = torch.Tensor(N, self.num_steps, *self.d_y).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        num_iter = self.num_steps
        steps_expanded = steps
        
        for k in range(num_iter):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, mean_final, var_final)
            t_old = x + gamma * gradx
            z = torch.randn(x.shape, device=x.device)
            x = t_old + torch.sqrt(2 * gamma)*z
            gradx = grad_gauss(x, mean_final, var_final)
            t_new = x + gamma * gradx
            x_tot[:, k, :] = x
            y_tot[:, k, :] = y
            out[:, k, :] = (t_old - t_new) #/ (2 * gamma)
            
        return x_tot, y_tot, out, steps_expanded

    def record_langevin_seq(self, net, init_samples_x, init_samples_y, sample=False):
        x = init_samples_x
        y = init_samples_y
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))

        
        x_tot = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        y_tot = torch.Tensor(N, self.num_steps, *self.d_y).to(x.device)
        out = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        steps_expanded = steps
        num_iter = self.num_steps
        
        if self.mean_match:
            for k in range(num_iter):
                gamma = self.gammas[k]    
                t_old = net(x, y, steps[:, k, :])
                
                if sample & (k==num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                    
                t_new = net(x, y, steps[:, k, :])
                x_tot[:, k, :] = x
                y_tot[:, k, :] = y
                out[:, k, :] = (t_old - t_new) 
        else:
            for k in range(num_iter):
                gamma = self.gammas[k]    
                t_old = x + net(x, y, steps[:, k, :])
                
                if sample & (k==num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                t_new = x + net(x, y, steps[:, k, :])
                
                x_tot[:, k, :] = x
                y_tot[:, k, :] = y
                out[:, k, :] = (t_old - t_new) 
            

        return x_tot, y_tot, out, steps_expanded

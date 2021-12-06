import torch
import os, sys
import torch.nn.functional as F
import numpy as np
from ..langevin import Langevin
from torch.utils.data import DataLoader
from .config_getters import get_models, get_optimizers, get_datasets, get_plotter, get_logger, get_tester
import datetime
from tqdm import tqdm
from .ema import EMAHelper
from . import repeater
import time
import random
import torch.autograd.profiler as profiler
from ..data import CacheLoader
from torch.utils.data import WeightedRandomSampler
from accelerate import Accelerator, DistributedType
import time

class IPFBase(torch.nn.Module):

    def __init__(self, init_ds, final_ds, mean_final, var_final, args):
        super().__init__()
        self.init_ds = init_ds
        self.final_ds = final_ds
        self.mean_final = mean_final
        self.var_final = var_final
        self.args = args
        
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(fp16=False, cpu=args.device=='cpu')
        self.device = self.accelerator.device # torch.device(args.device)

        # training params
        self.n_ipf =self.args.n_ipf
        self.num_steps =self.args.num_steps
        self.batch_size =self.args.batch_size
        self.num_iter =self.args.num_iter
        self.grad_clipping =self.args.grad_clipping
        self.fast_sampling =self.args.fast_sampling        
        self.lr =self.args.lr

        
        n = self.num_steps//2
        if self.args.gamma_space == 'linspace':
            gamma_half = np.linspace(self.args.gamma_min,args.gamma_max, n)
        elif self.args.gamma_space == 'geomspace':
            gamma_half = np.geomspace(self.args.gamma_min, self.args.gamma_max, n)
        self.gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        self.gammas = torch.tensor(self.gammas).to(self.device)
        self.T = torch.sum(self.gammas)

        # get models
        self.build_models()
        self.build_ema()

        # get optims
        self.build_optimizers()

        # get loggers
        self.logger = self.get_logger('train_logs')
        self.save_logger = self.get_logger('test_logs')

        # langevin
        if self.args.weight_distrib:
            alpha = self.args.weight_distrib_alpha
            prob_vec = (1 + alpha) * torch.sum(self.gammas) - torch.cumsum(self.gammas, 0)
        else:
            prob_vec = self.gammas * 0 + 1
        self.time_sampler = torch.distributions.categorical.Categorical(prob_vec)

        # get data
        self.build_dataloaders()

        # checkpoint
        date = str(datetime.datetime.now())[0:10]
        self.name_all = date

        # run from checkpoint
        self.checkpoint_run = self.args.checkpoint_run
        if self.args.checkpoint_run:
            self.checkpoint_it = self.args.checkpoint_it
            self.checkpoint_pass = self.args.checkpoint_pass
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'

        
        self.plotter = self.get_plotter()

        self.tester = self.get_tester()

        if self.accelerator.is_main_process:
            if not os.path.exists('./im'):
                os.mkdir('./im')
            if not os.path.exists('./gif'):
                os.mkdir('./gif')
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')


        self.stride = self.args.gif_stride
        self.stride_log = self.args.log_stride

        self.y_cond = self.args.y_cond
        if self.y_cond is not None:
            if isinstance(self.y_cond, str):
                self.y_cond = eval(self.y_cond).to(self.device)
            else:
                self.y_cond = list(self.y_cond)
                for j in range(len(self.y_cond)):
                    if isinstance(self.y_cond[j], str):
                        self.y_cond[j] = eval(self.y_cond[j]).to(self.device)
                self.y_cond = torch.stack(self.y_cond, dim=0)


    def get_logger(self, name='logs'):
        return get_logger(self.args, name)

    def get_plotter(self):
        return get_plotter(self, self.args)

    def get_tester(self):
        return get_tester(self, self.args)

    def build_models(self, forward_or_backward=None):
        # running network
        net_f, net_b = get_models(self.args)

        if self.args.checkpoint_run:
            if "checkpoint_f" in self.args:
                net_f.load_state_dict(torch.load(self.args.checkpoint_f))
            if "checkpoint_b" in self.args:
                net_b.load_state_dict(torch.load(self.args.checkpoint_b))                
        
        if self.args.dataparallel:
            net_f = torch.nn.DataParallel(net_f)
            net_b = torch.nn.DataParallel(net_b)

        if forward_or_backward is None:
            net_f = net_f.to(self.device)
            net_b = net_b.to(self.device)
            self.net = torch.nn.ModuleDict({'f': net_f, 'b': net_b})
        if forward_or_backward == 'f':
            net_f = net_f.to(self.device)
            self.net.update({'f': net_f})
        if forward_or_backward == 'b':
            net_b = net_b.to(self.device)
            self.net.update({'b': net_b})

        

    def accelerate(self, forward_or_backward):
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(self.net[forward_or_backward], self.optimizer[forward_or_backward])


    def update_ema(self, forward_or_backward):
        if self.args.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(mu=self.args.ema_rate, device=self.device)
            self.ema_helpers[forward_or_backward].register(self.net[forward_or_backward])


        
    def build_ema(self):
        if self.args.ema:
            self.ema_helpers={}                
            self.update_ema('f')
            self.update_ema('b')

            if self.args.checkpoint_run:
                # sample network
                sample_net_f, sample_net_b = get_models(self.args)
                
                if "sample_checkpoint_f" in self.args:
                    sample_net_f.load_state_dict(torch.load(self.args.sample_checkpoint_f))  
                    if self.args.dataparallel:
                        sample_net_f = torch.nn.DataParallel(sample_net_f)
                    sample_net_f=sample_net_f.to(self.device)
                    self.ema_helpers['f'].register(sample_net_f)
                if "sample_checkpoint_b" in self.args:
                    sample_net_b.load_state_dict(torch.load(self.args.sample_checkpoint_b))
                    if self.args.dataparallel:
                        sample_net_b = torch.nn.DataParallel(sample_net_b)
                    sample_net_b=sample_net_b.to(self.device)
                    self.ema_helpers['b'].register(sample_net_b)
                                      
    def build_optimizers(self):
        optimizer_f, optimizer_b = get_optimizers(self.net['f'], self.net['b'], self.lr)
        optimizer_b = optimizer_b
        optimizer_f = optimizer_f
        self.optimizer = {'f': optimizer_f, 'b':optimizer_b}

    def build_dataloaders(self):
        self.mean_final = self.mean_final.to(self.device)
        self.var_final = self.var_final.to(self.device)
        self.std_final = torch.sqrt(self.var_final).to(self.device)
        
        def worker_init_fn(worker_id):                                                          
            np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {"num_workers": self.args.num_workers, 
                       "pin_memory": self.args.pin_memory, 
                       "worker_init_fn": worker_init_fn,
                       'drop_last': True}

        self.save_npar = min(max(self.args.plot_npar, self.args.test_npar), len(self.init_ds))
        self.cache_npar = min(self.args.cache_npar, len(self.init_ds))
        
        # get plotter, gifs etc.
        self.save_init_dl = DataLoader(self.init_ds, batch_size=self.save_npar, shuffle=True, **self.kwargs)
        self.cache_init_dl = DataLoader(self.init_ds, batch_size=self.cache_npar, shuffle=True, **self.kwargs)
        (self.cache_init_dl, self.save_init_dl) = self.accelerator.prepare(self.cache_init_dl, self.save_init_dl)
        self.cache_init_dl = repeater(self.cache_init_dl)
        self.save_init_dl = repeater(self.save_init_dl)


        if self.args.transfer:
            self.save_final_dl = DataLoader(self.final_ds, batch_size=self.save_npar, shuffle=True, **self.kwargs)
            self.cache_final_dl = DataLoader(self.final_ds, batch_size=self.cache_npar, shuffle=True, **self.kwargs)
            (self.cache_final_dl, self.save_final_dl) = self.accelerator.prepare(self.cache_final_dl, self.save_final_dl)
            self.cache_final_dl = repeater(self.cache_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)
        else:
            self.cache_final_dl = None
            self.save_final = None

        batch = next(self.save_init_dl)
        batch_x = batch[0]
        batch_y = batch[1]
        shape_x = batch_x[0].shape
        shape_y = batch_y[0].shape
        self.shape_x = shape_x
        self.shape_y = shape_y

        self.langevin = Langevin(self.num_steps, shape_x, shape_y, self.gammas,
                                 self.time_sampler, device = self.device,
                                 mean_final=self.mean_final, var_final=self.var_final,
                                 mean_match=self.args.mean_match)
        
    def new_cacheloader(self, forward_or_backward, n, use_ema=True):
        
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(self.net[sample_direction])
        else:
            sample_net = self.net[sample_direction]
        
        if forward_or_backward == 'b':
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader('b',
                                sample_net, 
                                self.cache_init_dl, 
                                self.args.num_cache_batches, 
                                self.langevin, n, 
                                mean = None,
                                std = None,
                                batch_size=self.cache_npar,
                                device=self.device,
                                dataloader_f=self.cache_final_dl,
                                transfer=self.args.transfer)
            
        else: # forward
            sample_net = self.accelerator.prepare(sample_net)
            new_dl = CacheLoader('f',
                            sample_net, 
                            self.cache_init_dl, 
                            self.args.num_cache_batches, 
                            self.langevin, n, 
                            mean = self.mean_final,
                            std = self.std_final,
                            batch_size = self.cache_npar,
                            device=self.device,
                            dataloader_f=self.cache_final_dl,
                            transfer=self.args.transfer)
                                 

            
        new_dl = DataLoader(new_dl, batch_size=self.batch_size)

        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def train(self):
        pass

    def save_step(self,i, n, fb):
        if self.accelerator.is_main_process:
            if i % self.stride == 0:
                
                if self.args.ema:
                    sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
                else:
                    sample_net = self.net[fb]

                
                name_net =  'net' + '_' + fb +'_' + str(n) + "_" + str(i) + '.ckpt'
                name_net_ckpt = './checkpoints/' + name_net
                
                if self.args.dataparallel:
                    torch.save(self.net[fb].module.state_dict(), name_net_ckpt)
                else:
                    torch.save(self.net[fb].state_dict(), name_net_ckpt)
                    
                if self.args.ema:
                    name_net =  'sample_net' + '_' + fb +'_' + str(n) + "_" + str(i) + '.ckpt'
                    name_net_ckpt = './checkpoints/' + name_net
                    if self.args.dataparallel:
                        torch.save(sample_net.module.state_dict(), name_net_ckpt)
                    else:
                        torch.save(sample_net.state_dict(), name_net_ckpt)

                with torch.no_grad():
                    self.set_seed(seed=0 + self.accelerator.process_index)
                    if fb == 'f':
                        batch = next(self.save_init_dl)
                        batch_x = batch[0].to(self.device)
                        batch_y = batch[1].to(self.device)                                         
                    elif self.args.transfer:
                        batch = next(self.save_final_dl)[0]
                        batch = batch.to(self.device)
                    else:
                        batch_x = self.mean_final + self.std_final*torch.randn((self.save_npar, *self.shape_x), device=self.device)
                        init_batch = next(self.save_init_dl)
                        batch_y = init_batch[1].to(self.device)
                        init_batch_x = init_batch[0].to(self.device)
                        init_batch_y = batch_y                                       
                        
                    x_tot, y_tot, out, steps_expanded = self.langevin.record_langevin_seq(sample_net, batch_x, batch_y, sample=True)

                    shape_len = len(x_tot.shape)
                    x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
                    x_tot_plot = x_tot.detach()#.cpu().numpy()
                    y_tot = y_tot.permute(1, 0, *list(range(2, shape_len)))
                    y_tot_plot = y_tot.detach()#.cpu().numpy()

                    x_tot_cond = torch.zeros([0, *x_tot.shape])
                    x_tot_cond_fwdbwd = torch.zeros([0, *x_tot.shape])

                    if self.y_cond is not None and not self.args.transfer and fb == 'b':
                        for k in range(len(self.y_cond)):
                            x_tot_c = self.backward_sample(self.y_cond[k], final_batch_x=batch_x)
                            x_tot_cond = torch.cat([x_tot_cond, x_tot_c.cpu().unsqueeze(0)], dim=0)

                            x_tot_c_fwdbwd = self.forward_backward_sample(self.y_cond[k], init_batch_x, init_batch_y, n)
                            x_tot_cond_fwdbwd = torch.cat([x_tot_cond_fwdbwd, x_tot_c_fwdbwd.cpu().unsqueeze(0)], dim=0)

                test_metrics = self.tester(
                    batch_x[:self.args.test_npar], batch_y[:self.args.test_npar],
                    x_tot_plot[:, :self.args.test_npar], y_tot_plot[:, :self.args.test_npar],
                    x_tot_cond[:, :, :self.args.test_npar], self.y_cond,
                    self.args.data, self.save_init_dl, i, n, fb
                )
                test_metrics.update(self.tester.test_cond(self.y_cond, x_tot_cond_fwdbwd[:, :, :self.args.test_npar], 
                                                          self.args.data, i, n, fb, tag='fwdbwd'))
                test_metrics['T'] = self.T
                self.save_logger.log_metrics(test_metrics, step=i+self.num_iter*(n-1))
                
                self.plotter(batch_x[:self.args.plot_npar], x_tot_plot[:, :self.args.plot_npar], y_tot_plot[:, :self.args.plot_npar],
                             self.args.data, self.save_init_dl, self.y_cond, x_tot_cond[:, :, :self.args.plot_npar], i, n, fb)
                self.plotter.plot_sequence_cond(self.y_cond, x_tot_cond_fwdbwd[:, :self.args.plot_npar], 
                                                self.args.data, i, n, fb, tag='fwdbwd')

    def backward_sample(self, y_c, num_samples=None, final_batch_x=None, fix_seed=False):
        if self.accelerator.is_main_process:
            if self.args.ema:
                sample_net = self.ema_helpers['b'].ema_copy(self.net['b'])
            else:
                sample_net = self.net['b']

            with torch.no_grad():
                # self.set_seed(seed=0 + self.accelerator.process_index)
                if final_batch_x is not None:
                    final_batch_x = final_batch_x.to(self.device)
                else:
                    final_batch_x = self.mean_final + self.std_final * torch.randn((num_samples, *self.shape_x),
                                                                                   device=self.device)
                y_c = y_c.expand(final_batch_x.shape[0], *self.shape_y).clone().to(self.device)
                x_tot_c, _, _, _ = self.langevin.record_langevin_seq(sample_net, final_batch_x, y_c, sample=True)

                x_tot_c = x_tot_c.permute(1, 0, *list(range(2, len(x_tot_c.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot_c

    def forward_backward_sample(self, y_c, init_batch_x, init_batch_y, n, fix_seed=False):
        if self.accelerator.is_main_process:
            if self.args.ema:
                sample_net = self.ema_helpers['f'].ema_copy(self.net['f'])
            else:
                sample_net = self.net['f']

            with torch.no_grad():
                # self.set_seed(seed=0 + self.accelerator.process_index)
                init_batch_x = init_batch_x.to(self.device)
                init_batch_y = init_batch_y.to(self.device)
                if n == 1:
                    x_tot, _, _, _ = self.langevin.record_init_langevin(init_batch_x, init_batch_y)
                else:
                    x_tot, _, _, _ = self.langevin.record_langevin_seq(sample_net, init_batch_x, init_batch_y)

            final_batch_x = x_tot[:, -1]

        return self.backward_sample(y_c, final_batch_x=final_batch_x)
                
    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        torch.cuda.empty_cache()

class IPFSequential(IPFBase):
    def ipf_step(self, forward_or_backward, n):
        new_dl = None
        new_dl = self.new_cacheloader(forward_or_backward, n, self.args.ema)
        
        if not self.args.use_prev_net:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        self.build_optimizers() 
        self.accelerate(forward_or_backward)
        
        for i in tqdm(range(1, self.num_iter+1)):
            self.set_seed(seed=n*self.num_iter+i)

            x, y, out, steps_expanded = next(new_dl)
            x = x.to(self.device)
            y = y.to(self.device)
            out = out.to(self.device)
            steps_expanded = steps_expanded.to(self.device)
            eval_steps = self.num_steps - 1 - steps_expanded

            if self.args.mean_match:
                pred = self.net[forward_or_backward](x, y, eval_steps) - x
            else:
                pred = self.net[forward_or_backward](x, y, eval_steps) 

            loss = F.mse_loss(pred, out)
            
            #loss.backward()
            self.accelerator.backward(loss)
                

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = self.accelerator.clip_grad_norm_(self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.


            if i % self.stride_log == 0:
                self.logger.log_metrics({'forward_or_backward':forward_or_backward,
                                         'loss': loss, 
                                         'grad_norm' : total_norm}, step=i+self.num_iter*(n-1))
            
            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad()
            if self.args.ema:
                self.ema_helpers[forward_or_backward].update(self.net[forward_or_backward])
            

            self.save_step(i, n, forward_or_backward)
            
            if (i % self.args.cache_refresh_stride == 0) and (i != self.num_iter):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(forward_or_backward, n, self.args.ema)
        
        new_dl = None
        self.clear()
        

    def train(self):
        # INITIAL FORWARD PASS
        if self.accelerator.is_main_process:
            with torch.no_grad():
                self.set_seed(seed=0 + self.accelerator.process_index)
                batch = next(self.save_init_dl)
                batch_x = batch[0].to(self.device)
                batch_y = batch[1].to(self.device) 
                x_tot, y_tot, _, _ = self.langevin.record_init_langevin(batch_x, batch_y)
                shape_len = len(x_tot.shape)
                x_tot = x_tot.permute(1, 0, *list(range(2, shape_len)))
                x_tot_plot = x_tot.detach()#.cpu().numpy()
                y_tot = y_tot.permute(1, 0, *list(range(2, shape_len)))
                y_tot_plot = y_tot.detach()#.cpu().numpy()
                
                test_metrics = self.tester(
                        batch_x[:self.args.test_npar], batch_y[:self.args.test_npar],
                        x_tot_plot[:, :self.args.test_npar], y_tot_plot[:, :self.args.test_npar],
                        None, None, self.args.data, self.save_init_dl, 0, 0, 'f'
                    )
                test_metrics['T'] = self.T
                self.save_logger.log_metrics(test_metrics, step=0)
                
                self.plotter(batch_x[:self.args.plot_npar], x_tot_plot[:, :self.args.plot_npar], y_tot_plot[:, :self.args.plot_npar],
                                self.args.data, self.save_init_dl, None, None, 0, 0, 'f')

            x_tot_plot = None
            x_tot = None
            torch.cuda.empty_cache()
            
        for n in range(self.checkpoint_it, self.n_ipf+1):
            
            print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f',n)
            else:
                self.ipf_step('b',n)
                self.ipf_step('f',n)

    

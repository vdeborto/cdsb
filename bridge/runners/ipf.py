import torch
import os, sys, warnings
import torch.nn.functional as F
import numpy as np
from ..langevin import Langevin
from torch.utils.data import DataLoader
from .config_getters import get_models, get_optimizer, get_plotter, get_logger
import datetime
from tqdm import tqdm
from .ema import EMAHelper
from . import repeater
import time
import random
import torch.autograd.profiler as profiler
from ..data import CacheLoader
from torch.utils.data import TensorDataset
from accelerate import Accelerator, DistributedType

class IPFBase:
    def __init__(self, init_ds, final_ds, mean_final, var_final, args, accelerator=None, final_cond_model=None,
                 valid_ds=None, test_ds=None):
        super().__init__()
        if accelerator is None:
            self.accelerator = Accelerator(fp16=False, cpu=args.device == 'cpu', split_batches=True)
        else:
            self.accelerator = accelerator
        self.device = self.accelerator.device  # local device for each process

        self.args = args

        self.init_ds = init_ds
        self.final_ds = final_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.mean_final = mean_final
        self.var_final = var_final
        self.std_final = torch.sqrt(self.var_final) if self.var_final is not None else None

        self.transfer = self.args.transfer

        # training params
        self.n_ipf = self.args.n_ipf
        self.num_steps = self.args.num_steps
        self.batch_size = self.args.batch_size
        self.num_iter = self.args.num_iter
        self.grad_clipping = self.args.grad_clipping
        self.fast_sampling = self.args.fast_sampling
        self.lr = self.args.lr

        
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
        # self.build_optimizers()

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

        self.npar = len(init_ds)
        cache_epochs = (self.batch_size*self.args.cache_refresh_stride)/(self.cache_npar*self.args.num_cache_batches*self.num_steps)
        data_epochs = (self.num_iter*self.cache_npar*self.args.num_cache_batches)/(self.npar*self.args.cache_refresh_stride)
        self.accelerator.print("Cache epochs:", cache_epochs)
        self.accelerator.print("Data epochs:", data_epochs)
        if self.accelerator.is_main_process:
            if cache_epochs < 1:
                warnings.warn("Cache epochs < 1, increase batch_size, cache_refresh_stride, or decrease cache_npar, num_cache_batches, num_steps. ")
            if data_epochs < 1:
                warnings.warn("Data epochs < 1, increase num_iter, cache_npar, num_cache_batches, or decrease npar, cache_refresh_stride. ")

        # # checkpoint
        # date = str(datetime.datetime.now())[0:10]
        # self.name_all = date

        # run from checkpoint
        self.checkpoint_run = self.args.checkpoint_run
        if self.args.checkpoint_run:
            self.checkpoint_it = self.args.checkpoint_it
            self.checkpoint_pass = self.args.checkpoint_pass
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'

        self.plotter = self.get_plotter()
        if self.accelerator.is_main_process:

            ckpt_dir = './checkpoints/'
            os.makedirs(ckpt_dir, exist_ok=True)
            existing_versions = []
            for d in os.listdir(ckpt_dir):
                if os.path.isdir(os.path.join(ckpt_dir, d)) and d.startswith("version_"):
                    existing_versions.append(int(d.split("_")[1]))

            if len(existing_versions) == 0:
                version = 0
            else:
                version = max(existing_versions) + 1

            self.ckpt_dir = os.path.join(ckpt_dir, f"version_{version}")
            os.makedirs(self.ckpt_dir, exist_ok=True)


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
                    else:
                        self.y_cond[j] = torch.tensor([self.y_cond[j]]).to(self.device)

                self.y_cond = torch.stack(self.y_cond, dim=0)

        self.cond_final = self.args.cond_final
        assert (not self.transfer or not self.cond_final)
        if self.cond_final:
            self.final_cond_model = final_cond_model.to(self.device).eval()

    def get_logger(self, name='logs'):
        return get_logger(self.args, name)

    def get_plotter(self):
        return get_plotter(self, self.args)

    def build_models(self, forward_or_backward=None):
        # running network
        net_f, net_b = get_models(self.args)

        if self.args.checkpoint_run:
            if "checkpoint_f" in self.args:
                net_f.load_state_dict(torch.load(self.args.checkpoint_f))
            if "checkpoint_b" in self.args:
                net_b.load_state_dict(torch.load(self.args.checkpoint_b))                

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
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

            if self.args.checkpoint_run:
                # sample network
                sample_net_f, sample_net_b = get_models(self.args)
                
                if "sample_checkpoint_f" in self.args:
                    sample_net_f.load_state_dict(torch.load(self.args.sample_checkpoint_f))
                    sample_net_f = sample_net_f.to(self.device)
                    self.ema_helpers['f'].register(sample_net_f)
                if "sample_checkpoint_b" in self.args:
                    sample_net_b.load_state_dict(torch.load(self.args.sample_checkpoint_b))
                    sample_net_b = sample_net_b.to(self.device)
                    self.ema_helpers['b'].register(sample_net_b)
                                      
    def build_optimizer(self, forward_or_backward):
        optimizer = get_optimizer(self.net[forward_or_backward], self.lr)
        self.optimizer = {forward_or_backward: optimizer}

    def build_dataloaders(self):
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index)

        self.kwargs = {"num_workers": self.args.num_workers, 
                       "pin_memory": self.args.pin_memory, 
                       "worker_init_fn": worker_init_fn,
                       'drop_last': True}

        self.save_npar = min(max(self.args.plot_npar, self.args.test_npar), len(self.init_ds))
        self.cache_npar = min(self.args.cache_npar, len(self.init_ds))
        self.test_batch_size = min(self.save_npar, self.args.test_batch_size)

        self.cache_init_dl = DataLoader(self.init_ds, batch_size=self.cache_npar, shuffle=True, **self.kwargs)
        self.cache_init_dl = self.accelerator.prepare(self.cache_init_dl)
        self.cache_init_dl = repeater(self.cache_init_dl)

        self.save_init_dl = DataLoader(self.init_ds, batch_size=self.test_batch_size, **self.kwargs)
        self.save_init_dl = self.accelerator.prepare(self.save_init_dl)
        self.save_dls_dict = {"train": self.save_init_dl}

        if self.valid_ds is not None:
            self.save_valid_dl = DataLoader(self.valid_ds, batch_size=self.test_batch_size, **self.kwargs)
            self.save_valid_dl = self.accelerator.prepare(self.save_valid_dl)
            self.save_dls_dict["valid"] = self.save_valid_dl

        if self.test_ds is not None:
            self.save_test_dl = DataLoader(self.test_ds, batch_size=self.test_batch_size, **self.kwargs)
            self.save_test_dl = self.accelerator.prepare(self.save_test_dl)
            self.save_dls_dict["test"] = self.save_test_dl

        if self.transfer:
            self.cache_final_dl = DataLoader(self.final_ds, batch_size=self.cache_npar, shuffle=True, **self.kwargs)
            self.cache_final_dl = self.accelerator.prepare(self.cache_final_dl)
            self.cache_final_dl = repeater(self.cache_final_dl)

            self.save_final_dl = DataLoader(self.final_ds, batch_size=self.test_batch_size, shuffle=True, **self.kwargs)
            self.save_final_dl = self.accelerator.prepare(self.save_final_dl)
            self.save_final_dl = repeater(self.save_final_dl)
        else:
            self.cache_final_dl = None
            self.save_final_dl = None

        batch = next(self.cache_init_dl)
        batch_x = batch[0]
        batch_y = batch[1]
        shape_x = batch_x[0].shape
        shape_y = batch_y[0].shape
        self.shape_x = shape_x
        self.shape_y = shape_y

        self.langevin = Langevin(self.num_steps, shape_x, shape_y, self.gammas, self.time_sampler,
                                 mean_final=self.mean_final, var_final=self.var_final,
                                 mean_match=self.args.mean_match, out_scale=self.args.langevin_scale)

    def new_cacheloader(self, forward_or_backward, n, use_ema=True):
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
        if use_ema:
            sample_net = self.ema_helpers[sample_direction].ema_copy(self.net[sample_direction])
        else:
            sample_net = self.net[sample_direction]

        sample_net = sample_net.to(self.device)
        sample_net.eval()

        if forward_or_backward == 'b':
            new_ds = CacheLoader('b',
                                 sample_net,
                                 self.cache_init_dl,
                                 self.cache_final_dl,
                                 self.args.num_cache_batches,
                                 self.langevin, self, n,
                                 device='cpu' if self.args.cache_cpu else self.device)

        else:  # forward
            new_ds = CacheLoader('f',
                                 sample_net,
                                 self.cache_init_dl,
                                 self.cache_final_dl,
                                 self.args.num_cache_batches,
                                 self.langevin, self, n,
                                 device='cpu' if self.args.cache_cpu else self.device)

        new_dl = DataLoader(new_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def train(self):
        pass

    def save_step(self, i, n, fb):
        if i == 1 or i % self.stride == 0 or i == self.num_iter:
            if self.args.ema:
                sample_net = self.ema_helpers[fb].ema_copy(self.net[fb])
            else:
                sample_net = self.net[fb]

            if self.accelerator.is_main_process:
                name_net = 'net' + '_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
                torch.save(self.net[fb].state_dict(), name_net_ckpt)

                if self.args.ema:
                    name_net = 'sample_net' + '_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                    name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
                    torch.save(sample_net.state_dict(), name_net_ckpt)

            if not self.args.nosave:
                sample_net = sample_net.to(self.device)
                sample_net.eval()

                with torch.no_grad():
                    self.set_seed(seed=0 + self.accelerator.process_index)
                    test_metrics = self.plotter(sample_net, i, n, fb)

                    if self.accelerator.is_main_process:
                        self.save_logger.log_metrics(test_metrics, step=i+self.num_iter*(n-1))

    def sample_batch(self, init_dl, final_dl, fb, y_c=None):
        mean_final = self.mean_final
        std_final = self.std_final

        if fb == 'f':
            batch = next(init_dl)
            batch_x = batch[0]
            batch_y = batch[1]
            init_batch_x = batch_x

            if self.cond_final:
                mean, std = self.final_cond_model(batch_y)
                # batch_x = mean + std * torch.randn_like(init_batch_x)

                if self.args.final_adaptive:
                    mean_final = mean
                    std_final = std
                elif self.args.adaptive_mean:
                    mean_final = mean
        elif self.transfer:
            batch = next(final_dl)
            batch_x = batch[0]
            init_batch = next(init_dl)
            batch_y = init_batch[1]
            init_batch_x = init_batch[0]
        elif self.cond_final:
            init_batch = next(init_dl)
            init_batch_x = init_batch[0]
            batch_y = init_batch[1]
            if y_c is not None:
                batch_y = y_c.to(batch_y.device) + torch.zeros_like(batch_y)
                init_batch_x = None
            mean, std = self.final_cond_model(batch_y)
            batch_x = mean + std * torch.randn_like(init_batch_x)

            if self.args.final_adaptive:
                mean_final = mean
                std_final = std
            elif self.args.adaptive_mean:
                mean_final = mean
        else:
            init_batch = next(init_dl)
            init_batch_x = init_batch[0]
            batch_y = init_batch[1]
            mean_final = mean_final.to(init_batch_x.device)
            std_final = std_final.to(init_batch_x.device)
            batch_x = mean_final + std_final * torch.randn_like(init_batch_x)

        mean_final = mean_final.to(init_batch_x.device)
        std_final = std_final.to(init_batch_x.device)
        var_final = std_final ** 2
        return batch_x, batch_y, init_batch_x, mean_final, var_final

    def backward_sample(self, final_batch_x, y_c, fix_seed=False, sample_net=None):
        if sample_net is None:
            if self.args.ema:
                sample_net = self.ema_helpers['f'].ema_copy(self.net['f'])
            else:
                sample_net = self.net['f']

            sample_net = sample_net.to(self.device)
        sample_net.eval()

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            final_batch_x = final_batch_x.to(self.device)
            y_c = y_c.expand(final_batch_x.shape[0], *self.shape_y).clone().to(self.device)
            x_tot_c, _, _, _ = self.langevin.record_langevin_seq(sample_net, final_batch_x, y_c, sample=True)

            x_tot_c = x_tot_c.permute(1, 0, *list(range(2, len(x_tot_c.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot_c

    def forward_sample(self, init_batch_x, init_batch_y, n, fb, fix_seed=False, sample_net=None):
        if sample_net is None:
            if self.args.ema:
                sample_net = self.ema_helpers['f'].ema_copy(self.net['f'])
            else:
                sample_net = self.net['f']

            sample_net = sample_net.to(self.device)
        sample_net.eval()

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            init_batch_x = init_batch_x.to(self.device)
            init_batch_y = init_batch_y.to(self.device)
            if n == 1 and fb == "b":
                assert not self.cond_final
                mean_final = self.mean_final.to(self.device)
                var_final = self.var_final.to(self.device)

                x_tot, _, _, _ = self.langevin.record_init_langevin(init_batch_x, init_batch_y,
                                                                    mean_final=mean_final, var_final=var_final)
            else:
                x_tot, _, _, _ = self.langevin.record_langevin_seq(sample_net, init_batch_x, init_batch_y)

        x_tot = x_tot.permute(1, 0, *list(range(2, len(x_tot.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot

    def forward_backward_sample(self, init_batch_x, init_batch_y, y_c, n, fb, fix_seed=False, return_fwd_tot=False,
                                sample_net_f=None, sample_net_b=None):
        assert not self.cond_final

        x_tot = self.forward_sample(init_batch_x, init_batch_y, n, fb, fix_seed=fix_seed, sample_net=sample_net_f)
        final_batch_x = x_tot[-1]
        if return_fwd_tot:
            return x_tot, self.backward_sample(final_batch_x, y_c, fix_seed=fix_seed, sample_net=sample_net_b)
        return self.backward_sample(final_batch_x, y_c, fix_seed=fix_seed, sample_net=sample_net_b)

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        self.accelerator.free_memory()
        torch.cuda.empty_cache()


class IPFSequential(IPFBase):
    def ipf_step(self, forward_or_backward, n):
        new_dl = None
        new_dl = self.new_cacheloader(forward_or_backward, n, self.args.ema)
        
        if not self.args.use_prev_net:
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)

        self.build_optimizer(forward_or_backward)
        self.accelerate(forward_or_backward)
        
        for i in tqdm(range(1, self.num_iter+1)):
            self.net[forward_or_backward].train()

            self.set_seed(seed=n*self.num_iter+i)

            x, y, out, steps_expanded = next(new_dl)
            # x = x.to(self.device)
            # y = y.to(self.device)
            # out = out.to(self.device)
            # steps_expanded = steps_expanded.to(self.device)
            eval_steps = self.num_steps - 1 - steps_expanded

            if self.args.mean_match:
                pred = self.net[forward_or_backward](x, y, eval_steps) - x
            else:
                pred = self.net[forward_or_backward](x, y, eval_steps) 

            loss = F.mse_loss(pred, out)

            self.accelerator.backward(loss)
                

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = torch.nn.utils.clip_grad_norm_(self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.


            if i == 1 or i % self.stride_log == 0 or i == self.num_iter:
                self.logger.log_metrics({'fb': forward_or_backward,
                                         'ipf': n,
                                         'loss': loss,
                                         'grad_norm': total_norm}, step=i+self.num_iter*(n-1))
            
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

        self.net[forward_or_backward] = self.accelerator.unwrap_model(self.net[forward_or_backward])
        self.clear()

    def train(self):
        # INITIAL FORWARD PASS
        if not self.args.nosave:
            with torch.no_grad():
                self.set_seed(seed=0 + self.accelerator.process_index)
                test_metrics = self.plotter(None, 0, 0, 'f')

                if self.accelerator.is_main_process:
                    self.save_logger.log_metrics(test_metrics, step=0)

            
        for n in range(self.checkpoint_it, self.n_ipf+1):
            
            self.accelerator.print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                self.ipf_step('b', n)
                self.ipf_step('f', n)

    

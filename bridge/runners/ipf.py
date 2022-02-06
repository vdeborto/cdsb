import os, sys, warnings
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from ..langevin import Langevin
from torch.utils.data import DataLoader
from .config_getters import get_models, get_optimizer, get_plotter, get_logger
import hydra
from tqdm import tqdm
from .ema import EMAHelper
from . import repeater
import time
import random
import torch.autograd.profiler as profiler
from ..data import CacheLoader
from bridge.runners.accelerator import Accelerator


class IPFBase:
    def __init__(self, init_ds, final_ds, mean_final, var_final, args, accelerator=None, final_cond_model=None,
                 valid_ds=None, test_ds=None):
        super().__init__()
        if accelerator is None:
            self.accelerator = Accelerator(train_batch_size=args.batch_size, cpu=args.device == 'cpu',
                                           fp16=args.model.use_fp16, split_batches=True)
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

        if self.args.symmetric_gamma:
            n = self.num_steps // 2
            if self.args.gamma_space == 'linspace':
                gamma_half = np.linspace(self.args.gamma_min, self.args.gamma_max, n)
            elif self.args.gamma_space == 'geomspace':
                gamma_half = np.geomspace(self.args.gamma_min, self.args.gamma_max, n)
            self.gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
        else:
            if self.args.gamma_space == 'linspace':
                self.gammas = np.linspace(self.args.gamma_min, self.args.gamma_max, self.num_steps)
            elif self.args.gamma_space == 'geomspace':
                self.gammas = np.geomspace(self.args.gamma_min, self.args.gamma_max, self.num_steps)
        self.gammas = torch.tensor(self.gammas).to(self.device)
        self.T = torch.sum(self.gammas)
        self.accelerator.print("T:", self.T.item())

        # get models
        self.first_pass = True
        self.build_models()
        self.build_ema()

        # get optims
        self.build_optimizers()

        # get loggers
        self.logger = self.get_logger('train_logs')
        self.save_logger = self.get_logger('test_logs')

        # langevin
        # if self.args.weight_distrib:
        #     alpha = self.args.weight_distrib_alpha
        #     prob_vec = (1 + alpha) * torch.sum(self.gammas) - torch.cumsum(self.gammas, 0)
        # else:
        #     prob_vec = self.gammas * 0 + 1
        self.time_sampler = None  # torch.distributions.categorical.Categorical(prob_vec)

        # get data
        self.build_dataloaders()

        self.npar = len(init_ds)
        self.cache_epochs = (self.batch_size * self.args.cache_refresh_stride) / (
                        self.cache_npar * self.args.num_cache_batches * self.num_steps)
        self.data_epochs = (self.num_iter * self.cache_npar * self.args.num_cache_batches) / (
                        self.npar * self.args.cache_refresh_stride)
        self.accelerator.print("Cache epochs:", self.cache_epochs)
        self.accelerator.print("Data epochs:", self.data_epochs)
        if self.accelerator.is_main_process:
            if self.cache_epochs < 1:
                warnings.warn(
                    "Cache epochs < 1, increase batch_size, cache_refresh_stride, or decrease cache_npar, num_cache_batches, num_steps. ")
            if self.data_epochs < 1:
                warnings.warn(
                    "Data epochs < 1, increase num_iter, cache_npar, num_cache_batches, or decrease npar, cache_refresh_stride. ")

        # # checkpoint
        # date = str(datetime.datetime.now())[0:10]
        # self.name_all = date

        # run from checkpoint
        self.checkpoint_run = self.args.checkpoint_run
        if self.args.checkpoint_run:
            self.checkpoint_it = self.args.checkpoint_it
            self.checkpoint_pass = self.args.checkpoint_pass
            self.checkpoint_iter = self.args.checkpoint_iter + 1
        else:
            self.checkpoint_it = 1
            self.checkpoint_pass = 'b'
            self.checkpoint_iter = 1

        if not self.args.nosave:
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

        self.x_cond_true = self.args.x_cond_true
        if self.x_cond_true is not None:
            if isinstance(self.x_cond_true, str):
                self.x_cond_true = eval(self.x_cond_true).to(self.device)
            else:
                self.x_cond_true = list(self.x_cond_true)
                for j in range(len(self.x_cond_true)):
                    if isinstance(self.x_cond_true[j], str):
                        self.x_cond_true[j] = eval(self.x_cond_true[j]).to(self.device)
                    else:
                        self.x_cond_true[j] = torch.tensor([self.x_cond_true[j]]).to(self.device)

                self.x_cond_true = torch.stack(self.x_cond_true, dim=0)

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

        if self.first_pass and self.args.checkpoint_run:
            if self.args.checkpoint_f is not None:
                try:
                    net_f.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_f)))
                except:
                    state_dict = torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_f))
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")  # remove "module."
                        new_state_dict[name] = v
                    net_f.load_state_dict(new_state_dict)

            if self.args.checkpoint_b is not None:
                try:
                    net_b.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_b)))
                except:
                    state_dict = torch.load(hydra.utils.to_absolute_path(self.args.checkpoint_b))
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")  # remove "module."
                        new_state_dict[name] = v
                    net_b.load_state_dict(new_state_dict)

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
        (self.net[forward_or_backward], self.optimizer[forward_or_backward]) = self.accelerator.prepare(
            self.net[forward_or_backward], self.optimizer[forward_or_backward])

    def update_ema(self, forward_or_backward):
        if self.args.ema:
            self.ema_helpers[forward_or_backward] = EMAHelper(mu=self.args.ema_rate, device=self.device)
            self.ema_helpers[forward_or_backward].register(self.accelerator.unwrap_model(self.net[forward_or_backward]))

    def build_ema(self):
        if self.args.ema:
            self.ema_helpers = {}
            self.update_ema('f')
            self.update_ema('b')

            if self.first_pass and self.args.checkpoint_run:
                # sample network
                sample_net_f, sample_net_b = get_models(self.args)

                if self.args.sample_checkpoint_f is not None:
                    sample_net_f.load_state_dict(
                        torch.load(hydra.utils.to_absolute_path(self.args.sample_checkpoint_f)))
                    sample_net_f = sample_net_f.to(self.device)
                    self.ema_helpers['f'].register(sample_net_f)
                if self.args.sample_checkpoint_b is not None:
                    sample_net_b.load_state_dict(
                        torch.load(hydra.utils.to_absolute_path(self.args.sample_checkpoint_b)))
                    sample_net_b = sample_net_b.to(self.device)
                    self.ema_helpers['b'].register(sample_net_b)

    def build_optimizers(self, forward_or_backward=None):
        pass

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
                                 mean_match=self.args.mean_match, out_scale=self.args.langevin_scale,
                                 var_final_gamma_scale=self.args.var_final_gamma_scale,
                                 double_gamma_scale=self.args.double_gamma_scale)

    def get_sample_net(self, fb):
        if self.args.ema:
            sample_net = self.ema_helpers[fb].ema_copy(self.accelerator.unwrap_model(self.net[fb]))
        else:
            sample_net = self.net[fb]
        return sample_net

    def new_cacheloader(self, forward_or_backward, n):
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
        sample_net = self.get_sample_net(sample_direction)
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

        assert self.batch_size % self.accelerator.num_processes == 0
        new_dl = DataLoader(new_ds, batch_size=self.batch_size // self.accelerator.num_processes, shuffle=True,
                            drop_last=True, pin_memory=self.args.pin_memory)

        # new_dl = self.accelerator.prepare(new_dl)
        new_dl = repeater(new_dl)
        return new_dl

    def ipf_step(self, forward_or_backward, n):
        pass

    def train(self):
        # INITIAL FORWARD PASS
        if not self.args.nosave and not self.args.checkpoint_run:
            with torch.no_grad():
                self.set_seed(seed=0 + self.accelerator.process_index)
                test_metrics = self.plotter(None, 0, 0, 'f')

                if self.accelerator.is_main_process:
                    self.save_logger.log_metrics(test_metrics, step=0)

        for n in range(self.checkpoint_it, self.n_ipf + 1):

            self.accelerator.print('IPF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            if (self.checkpoint_pass == 'f') and (n == self.checkpoint_it):
                self.ipf_step('f', n)
            else:
                self.ipf_step('b', n)
                self.ipf_step('f', n)

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

    def backward_sample(self, final_batch_x, y_c, fix_seed=False, sample_net=None, var_final=None):
        if sample_net is None:
            sample_net = self.get_sample_net('b')
            sample_net = sample_net.to(self.device)
        sample_net.eval()

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            final_batch_x = final_batch_x.to(self.device)
            y_c = y_c.expand(final_batch_x.shape[0], *self.shape_y).clone().to(self.device)
            x_tot_c, _, _, _ = self.langevin.record_langevin_seq(sample_net, final_batch_x, y_c, 'b', sample=True,
                                                                 var_final=var_final)

            x_tot_c = x_tot_c.permute(1, 0, *list(range(2, len(x_tot_c.shape))))  # (num_steps, num_samples, *shape_x)

        return x_tot_c

    def forward_sample(self, init_batch_x, init_batch_y, n, fb, fix_seed=False, sample_net=None):
        if sample_net is None:
            sample_net = self.get_sample_net('f')
            sample_net = sample_net.to(self.device)
        sample_net.eval()

        with torch.no_grad():
            # self.set_seed(seed=0 + self.accelerator.process_index)
            init_batch_x = init_batch_x.to(self.device)
            init_batch_y = init_batch_y.to(self.device)
            assert not self.cond_final
            mean_final = self.mean_final.to(self.device)
            var_final = self.var_final.to(self.device)
            if n == 1 and fb == "b":

                x_tot, _, _, _ = self.langevin.record_init_langevin(init_batch_x, init_batch_y,
                                                                    mean_final=mean_final, var_final=var_final)
            else:
                x_tot, _, _, _ = self.langevin.record_langevin_seq(sample_net, init_batch_x, init_batch_y, 'f',
                                                                   var_final=var_final)

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
    def build_optimizers(self, forward_or_backward=None):
        optimizer_f, optimizer_b = get_optimizer(self.net['f'], self.args), get_optimizer(self.net['b'], self.args)

        if self.first_pass and self.args.checkpoint_run:
            if self.args.optimizer_checkpoint_f is not None:
                optimizer_f.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_f)))
            if self.args.optimizer_checkpoint_b is not None:
                optimizer_b.load_state_dict(torch.load(hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_b)))

        if forward_or_backward is None:
            self.optimizer = {'f': optimizer_f, 'b': optimizer_b}
        if forward_or_backward == 'f':
            self.optimizer.update({'f': optimizer_f})
        if forward_or_backward == 'b':
            self.optimizer.update({'b': optimizer_b})

    def save_step(self, i, n, fb):
        if (self.first_pass and i == 1) or i % self.stride == 0 or i == self.num_iter:
            sample_net = self.get_sample_net(fb)

            if self.accelerator.is_main_process:
                name_net = 'net' + '_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
                torch.save(self.accelerator.unwrap_model(self.net[fb]).state_dict(), name_net_ckpt)
                name_opt = 'optimizer' + '_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                name_opt_ckpt = os.path.join(self.ckpt_dir, name_opt)
                torch.save(self.optimizer[fb].optimizer.state_dict(), name_opt_ckpt)
                # if self.args.LOGGER == 'Wandb':
                #     import wandb
                #     wandb.save(name_net_ckpt)

                if self.args.ema:
                    name_net = 'sample_net' + '_' + fb + '_' + str(n) + "_" + str(i) + '.ckpt'
                    name_net_ckpt = os.path.join(self.ckpt_dir, name_net)
                    torch.save(sample_net.state_dict(), name_net_ckpt)
                    # if self.args.LOGGER == 'Wandb':
                    #     import wandb
                    #     wandb.save(name_net_ckpt)

            if not self.args.nosave:
                sample_net = sample_net.to(self.device)
                sample_net.eval()

                with torch.no_grad():
                    self.set_seed(seed=0 + self.accelerator.process_index)
                    test_metrics = self.plotter(sample_net, i, n, fb)

                    if self.accelerator.is_main_process:
                        self.save_logger.log_metrics(test_metrics, step=i + self.num_iter * (n - 1))

    def ipf_step(self, forward_or_backward, n):
        new_dl = self.new_cacheloader(forward_or_backward, n)

        if (not self.first_pass) and (not self.args.use_prev_net):
            self.build_models(forward_or_backward)
            self.update_ema(forward_or_backward)
            self.build_optimizers(forward_or_backward)

        self.accelerate(forward_or_backward)

        if self.first_pass:
            checkpoint_iter = self.checkpoint_iter
        else:
            checkpoint_iter = 1

        for i in tqdm(range(checkpoint_iter, self.num_iter + 1)):
            self.net[forward_or_backward].train()

            self.set_seed(seed=n * self.num_iter + i + self.accelerator.process_index)

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

            self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = self.accelerator.clip_grad_norm_(self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.

            if i == 1 or i % self.stride_log == 0 or i == self.num_iter:
                self.logger.log_metrics({'fb': forward_or_backward,
                                         'ipf': n,
                                         'loss': loss,
                                         'grad_norm': total_norm,
                                         "cache_epochs": self.cache_epochs,
                                         "data_epochs": self.data_epochs}, step=i + self.num_iter * (n - 1))

            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad(set_to_none=True)
            if self.args.ema:
                self.ema_helpers[forward_or_backward].update(self.accelerator.unwrap_model(self.net[forward_or_backward]))

            self.save_step(i, n, forward_or_backward)

            if (i % self.args.cache_refresh_stride == 0) and (i != self.num_iter):
                new_dl = None
                torch.cuda.empty_cache()
                new_dl = self.new_cacheloader(forward_or_backward, n)

        new_dl = None

        self.net[forward_or_backward] = self.accelerator.unwrap_model(self.net[forward_or_backward])
        self.clear()
        self.first_pass = False


class IPFAnalytic(IPFBase):
    def new_cacheloader(self, forward_or_backward, n):
        sample_direction = 'f' if forward_or_backward == 'b' else 'b'
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

        return new_ds

    def ipf_step(self, forward_or_backward, n):
        new_dl = self.new_cacheloader(forward_or_backward, n)

        if not self.args.use_prev_net:
            self.build_models(forward_or_backward)

        x, y, out, steps_expanded = new_dl.tensors
        eval_steps = self.num_steps - 1 - steps_expanded

        if self.args.mean_match:
            self.net[forward_or_backward].fit(x, y, out + x, eval_steps)
        else:
            self.net[forward_or_backward].fit(x, y, out, eval_steps)

        if not self.args.nosave:
            with torch.no_grad():
                self.set_seed(seed=0 + self.accelerator.process_index)
                test_metrics = self.plotter(self.net[forward_or_backward], 1, n, forward_or_backward)

                if self.accelerator.is_main_process:
                    self.save_logger.log_metrics(test_metrics, step=0)

        self.first_pass = False

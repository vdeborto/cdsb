import torch
from omegaconf import OmegaConf
import hydra
from ..models import *
# from ..data.biochemical import biochemical_ds
from ..data.one_dim_cond import one_dim_cond_ds
from ..data.five_dim_cond import five_dim_cond_ds
from ..data.lorenz import lorenz_process, lorenz_ds
from ..data.stackedmnist import Cond_Stacked_MNIST
from ..data.emnist import EMNIST
from ..data.celeba import Cond_CelebA
from .plotters import Plotter, OneDCondPlotter, FiveDCondPlotter, BiochemicalPlotter, ImPlotter
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import os
from .logger import CSVLogger, WandbLogger, Logger
from torch.utils.data import DataLoader
cmp = lambda x: transforms.Compose([*x])


def get_plotter(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag == DATASET_1D_COND:
        return OneDCondPlotter(runner, args)
    elif dataset_tag == DATASET_5D_COND:
        return FiveDCondPlotter(runner, args)
    elif dataset_tag == DATASET_BIOCHEMICAL:
        return BiochemicalPlotter(runner, args)
    elif dataset_tag in [DATASET_STACKEDMNIST, DATASET_CELEBA]:
        return ImPlotter(runner, args)
    else:
        return Plotter(runner, args)


# Model
#--------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL_COND = 'BasicCond'
SUPERRES_UNET_MODEL = 'SuperResUNET'


def get_models(args):
    model_tag = getattr(args, MODEL)

    if model_tag == BASIC_MODEL_COND:
        x_dim = args.x_dim
        y_dim = args.y_dim

        kwargs = {
                    "encoder_layers": args.model.encoder_layers,
                    "temb_dim": args.model.temb_dim,
                    "decoder_layers": args.model.decoder_layers,
                    "temb_denom": args.model.temb_denom
                }
        net_f, net_b = ScoreNetworkCond(x_dim=x_dim, y_dim=y_dim, **kwargs), ScoreNetworkCond(x_dim=x_dim, y_dim=y_dim, **kwargs)

    if model_tag == SUPERRES_UNET_MODEL:
        image_size=args.data.image_size

        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 28:
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in args.model.attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        
        kwargs = {
                    "in_channels": args.data.channels,
                    "model_channels": args.model.num_channels,
                    "out_channels": args.data.channels,
                    "num_res_blocks": args.model.num_res_blocks,
                    "attention_resolutions": tuple(attention_ds),
                    "dropout": args.model.dropout,
                    "channel_mult": channel_mult,
                    "num_classes": None,
                    "use_checkpoint": args.model.use_checkpoint,
                    "num_heads": args.model.num_heads,
                    "num_heads_upsample": args.model.num_heads_upsample,
                    "use_scale_shift_norm": args.model.use_scale_shift_norm
                }

        net_f, net_b = SuperResModel(**kwargs), SuperResModel(**kwargs)

    return net_f, net_b


def get_final_cond_model(args, init_ds):
    assert args.cond_final

    model_tag = args.cond_final_model.MODEL

    if model_tag == 'BasicCond':
        mean_scale = args.cond_final_model.mean_scale
        if args.cond_final_model.adaptive_std:
            data_x, data_y = init_ds[:len(init_ds)]
            std = torch.std(data_x - data_y).item() * args.cond_final_model.std_scale
        else:
            std = args.cond_final_model.std_scale
        print("Final cond model std:", std)
        final_cond_model = BasicCondGaussian(mean_scale, std)

    return final_cond_model

# Optimizer
#--------------------------------------------------------------------------------

def get_optimizer(net, lr):
    return torch.optim.Adam(net.parameters(), lr=lr)


# Dataset
#--------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_TRANSFER = 'Dataset_transfer'
DATASET_1D_COND = '1d_cond'
DATASET_5D_COND = '5d_cond'
DATASET_BIOCHEMICAL = 'biochemical'
DATASET_LORENZ = 'lorenz'
DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'
DATASET_EMNIST = 'emnist'

NAPPROX = 2000

def get_datasets(args):
    dataset_tag = getattr(args, DATASET)

    # INITIAL (DATA) DATASET

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    # BIOCHEMICAL

    if dataset_tag == DATASET_BIOCHEMICAL:
        data_tag = args.data.dataset
        npar = args.npar
        init_ds = biochemical_ds(npar, data_tag)
        
    # 1D CONDITIONAL DATASET        
    
    if dataset_tag == DATASET_1D_COND:
        assert args.x_dim == 1
        assert args.y_dim == 1
        data_tag = args.data.dataset
        npar = args.npar
        root = os.path.join(data_dir, '1d_cond')
        init_ds = one_dim_cond_ds(root, npar, data_tag)    
    
    # 5D CONDITIONAL DATASET
    
    if dataset_tag == DATASET_5D_COND:
        assert args.x_dim == 1
        assert args.y_dim == 5
        data_tag = args.data.dataset
        npar = args.npar
        root = os.path.join(data_dir, '5d_cond')
        init_ds = five_dim_cond_ds(root, npar, data_tag)

    # CELEBA DATASET

    if dataset_tag == DATASET_CELEBA:

        train_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size),
                           transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if args.data.random_flip:
            train_transform.insert(2, transforms.RandomHorizontalFlip())


        data_tag = args.data.dataset
        root = data_dir
        init_ds = Cond_CelebA(data_tag, root, split='train', transform=cmp(train_transform), download=False)

    # MNIST DATASET

    if dataset_tag == DATASET_STACKEDMNIST:
        data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        init_ds = Cond_Stacked_MNIST(data_tag, root=root, load=load, split='train', num_channels=args.data.channels)

    # EMNIST DATASET

    if dataset_tag == DATASET_EMNIST:
        root = os.path.join(data_dir, 'EMNIST')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        init_ds = EMNIST(root, load=load, source_root=root,
                                train=True, num_channels = args.data.channels,
                                imageSize=args.data.image_size)

    # FINAL (GAUSSIAN) DATASET (if no transfer)

    final_ds, mean_final, var_final = get_final_dataset(args, init_ds)
    return init_ds, final_ds, mean_final, var_final


def get_final_dataset(args, init_ds):
    dataset_tag = getattr(args, DATASET)
    if args.transfer:
        dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
    else:
        dataset_transfer_tag = None

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)


    # if dataset_transfer_tag == DATASET_STACKEDMNIST:
    #     root = os.path.join(data_dir, 'mnist')
    #     saved_file = os.path.join(root, "data.pt")
    #     load = os.path.exists(saved_file)
    #     load = args.load
    #     final_ds = Cond_Stacked_MNIST(args, root=root, load=load, source_root=root,
    #                                   train=True, num_channels=args.data.channels,
    #                                   imageSize=args.data.image_size,
    #                                   device=args.device)
    #     mean_final = torch.tensor(0.)
    #     var_final = torch.tensor(1. * 10 ** 3)
    #
    # # EMNIST DATASET
    #
    # if dataset_transfer_tag == DATASET_EMNIST:
    #     root = os.path.join(data_dir, 'EMNIST')
    #     saved_file = os.path.join(root, "data.pt")
    #     load = os.path.exists(saved_file)
    #     load = args.load
    #     final_ds = EMNIST(root, load=load, source_root=root,
    #                       train=True, num_channels=args.data.channels,
    #                       imageSize=args.data.image_size,
    #                       device=args.device)
    #     mean_final = torch.tensor(0.)
    #     var_final = torch.tensor(1. * 10 ** 3)


    # FINAL (GAUSSIAN) DATASET (if no transfer)
    if not args.transfer:
        if args.adaptive_mean:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, shuffle=True)))[0]
            mean_final = vec.mean(axis=0)
            # mean_final = vec[0] * 0 + mean_final
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        elif args.final_adaptive:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, shuffle=True)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var(axis=0)
        else:
            mean_final = eval(args.mean_final) if isinstance(args.mean_final, str) else torch.tensor([args.mean_final])
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        final_ds = None

    return final_ds, mean_final, var_final


def get_valid_test_datasets(args):
    valid_ds, test_ds = None, None

    dataset_tag = getattr(args, DATASET)
    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    # MNIST DATASET

    if dataset_tag == DATASET_STACKEDMNIST:
        data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        valid_ds = Cond_Stacked_MNIST(data_tag, root=root, load=load, split='valid', num_channels=args.data.channels)
        test_ds = Cond_Stacked_MNIST(data_tag, root=root, load=load, split='test', num_channels=args.data.channels)

    # CELEBA DATASET

    if dataset_tag == DATASET_CELEBA:
        test_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size),
                          transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        data_tag = args.data.dataset
        root = data_dir
        valid_ds = Cond_CelebA(data_tag, root, split='valid', transform=cmp(test_transform), download=False)
        test_ds = Cond_CelebA(data_tag, root, split='test', transform=cmp(test_transform), download=False)

    return valid_ds, test_ds


def get_filtering_process(args):
    dataset_tag = getattr(args, DATASET)

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    if dataset_tag == DATASET_LORENZ:
        data_tag = args.data.dataset
        root = os.path.join(data_dir, 'lorenz')
        x, y, gt_means, gt_stds = lorenz_process(root, data_tag)

    return x, y, gt_means, gt_stds


def get_filtering_datasets(x_tm1, args):
    dataset_tag = getattr(args, DATASET)

    if dataset_tag == DATASET_LORENZ:
        data_tag = args.data.dataset
        init_ds = lorenz_ds(x_tm1, data_tag)

    if args.transfer:
        if dataset_tag == DATASET_LORENZ:
            init_x, init_y = init_ds.tensors
            npar = init_x.shape[0]
            final_ds = TensorDataset(init_x[torch.randperm(npar)], init_y[torch.randperm(npar)])

        if args.adaptive_mean:
            vec = next(iter(DataLoader(final_ds, batch_size=NAPPROX, shuffle=True)))[0]
            mean_final = vec.mean(axis=0)
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        elif args.final_adaptive:
            vec = next(iter(DataLoader(final_ds, batch_size=NAPPROX, shuffle=True)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var(axis=0)
        else:
            mean_final = eval(args.mean_final) if isinstance(args.mean_final, str) else torch.tensor([args.mean_final])
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])

    if not args.transfer:
        if args.adaptive_mean:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, shuffle=True)))[0]
            mean_final = vec.mean(axis=0)
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        elif args.final_adaptive:
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, shuffle=True)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var(axis=0)
        else:
            mean_final = eval(args.mean_final) if isinstance(args.mean_final, str) else torch.tensor([args.mean_final])
            var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
        final_ds = None

    return init_ds, final_ds, mean_final, var_final


# Logger
#--------------------------------------------------------------------------------

LOGGER = 'LOGGER'
LOGGER_PARAMS = 'LOGGER_PARAMS'

CSV_TAG = 'CSV'
WANDB_TAG = 'Wandb'
NOLOG_TAG = 'NONE'

def get_logger(args, name):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {'save_dir': args.CSV_log_dir, 'name': name, 'flush_logs_every_n_steps': 1}
        return CSVLogger(**kwargs)

    if logger_tag == WANDB_TAG:
        log_dir = os.getcwd()
        run_name = os.path.normpath(os.path.relpath(log_dir, hydra.utils.to_absolute_path(args.paths.experiments_dir_name))).replace("\\", "/")
        data_tag = args.data.dataset
        config = OmegaConf.to_container(args, resolve=True)

        kwargs = {'name': run_name, 'project': 'cdsb_develop_' + args.Dataset, 'prefix': name, 'tags': [data_tag], 'config': config}
        return WandbLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()

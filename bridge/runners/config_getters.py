import torch
import hydra
from ..models import *
from ..data.biochemical import biochemical_ds
from ..data.one_dim_cond import one_dim_cond_ds
from ..data.five_dim_cond import five_dim_cond_ds
from ..data.lorenz import lorenz_process, lorenz_ds
from ..data.two_dim import two_dim_ds
from ..data.stackedmnist import Stacked_MNIST
from ..data.emnist import EMNIST
from ..data.celeba  import CelebA
from .plotters import Plotter, OneDCondPlotter, FiveDCondPlotter, BiochemicalPlotter, TwoDPlotter, ImPlotter
from .testers import Tester, OneDCondTester, FiveDCondTester
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import os
from .logger import CSVLogger, NeptuneLogger, Logger
from torch.utils.data import DataLoader
cmp = lambda x: transforms.Compose([*x])

def get_plotter(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag == DATASET_1D_COND:
        return OneDCondPlotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)
    elif dataset_tag == DATASET_5D_COND:
        return FiveDCondPlotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)
    elif dataset_tag == DATASET_BIOCHEMICAL:
        return BiochemicalPlotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)
    elif dataset_tag == DATASET_2D:
        return TwoDPlotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)
    else:
        return Plotter(num_steps=runner.num_steps, gammas=runner.langevin.gammas)

def get_tester(runner, args):
    dataset_tag = getattr(args, DATASET)
    if dataset_tag == DATASET_1D_COND:
        return OneDCondTester()
    elif dataset_tag == DATASET_5D_COND:
        return FiveDCondTester()
    else:
        return Tester()

# Model
#--------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
BASIC_MODEL_COND = 'BasicCond'
UNET_MODEL = 'UNET'


def get_models(args):
    x_dim = args.x_dim
    y_dim = args.y_dim
        
    model_tag = getattr(args, MODEL)

    if model_tag == BASIC_MODEL_COND:
        kwargs = {
                    "encoder_layers": args.model.encoder_layers,
                    "temb_dim": args.model.temb_dim,
                    "decoder_layers": args.model.decoder_layers,
                    "temb_denom": args.model.temb_denom
                }
        net_f, net_b = ScoreNetworkCond(x_dim=x_dim, y_dim=y_dim, **kwargs), ScoreNetworkCond(x_dim=x_dim, y_dim=y_dim, **kwargs)
    
    if model_tag == BASIC_MODEL:
        net_f, net_b = ScoreNetwork(), ScoreNetwork()

    if model_tag == UNET_MODEL:
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

        net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)

    return net_f, net_b

# Optimizer
#--------------------------------------------------------------------------------
def get_optimizers(net_f, net_b, lr):
    return torch.optim.Adam(net_f.parameters(), lr=lr), torch.optim.Adam(net_b.parameters(), lr=lr)

# Dataset
#--------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_TRANSFER = 'Dataset_transfer'
DATASET_1D_COND = '1d_cond'
DATASET_5D_COND = '5d_cond'
DATASET_BIOCHEMICAL = 'biochemical'
DATASET_2D = '2d'
DATASET_LORENZ = 'lorenz'
DATASET_CELEBA = 'celeba'
DATASET_STACKEDMNIST = 'stackedmnist'
DATASET_EMNIST = 'emnist'


def get_datasets(args):
    dataset_tag = getattr(args, DATASET)
    if args.transfer:
        dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
    else:
        dataset_transfer_tag = None
    
    # INITIAL (DATA) DATASET

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    # BIOCHEMICAL

    if dataset_tag == DATASET_BIOCHEMICAL:
        data_tag = args.data
        npar = args.npar
        init_ds = biochemical_ds(npar, data_tag)
        
    # 1D CONDITIONAL DATASET        
    
    if dataset_tag == DATASET_1D_COND:
        assert args.x_dim == 1
        assert args.y_dim == 1
        data_tag = args.data
        npar = args.npar
        root = os.path.join(data_dir, '1d_cond')
        init_ds = one_dim_cond_ds(root, npar, data_tag)    
    
    # 5D CONDITIONAL DATASET
    
    if dataset_tag == DATASET_5D_COND:
        assert args.x_dim == 1
        assert args.y_dim == 5
        data_tag = args.data
        npar = args.npar
        root = os.path.join(data_dir, '5d_cond')
        init_ds = five_dim_cond_ds(root, npar, data_tag)

    # 2D DATASET
    
    if dataset_tag == DATASET_2D:
        data_tag = args.data
        npar = args.npar
        init_ds = two_dim_ds(npar, data_tag)

    if dataset_transfer_tag == DATASET_2D:
        data_tag = args.data_transfer
        npar = args.npar
        final_ds = two_dim_ds(npar, data_tag)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3) #infty like

    # CELEBA DATASET

    if dataset_tag == DATASET_CELEBA:

        train_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size), transforms.ToTensor()]
        test_transform = [transforms.CenterCrop(140), transforms.Resize(args.data.image_size), transforms.ToTensor()]
        if args.data.random_flip:
            train_transform.insert(2, transforms.RandomHorizontalFlip())


        root = os.path.join(data_dir, 'celeba')
        init_ds = CelebA(root, split='train', transform=cmp(train_transform), download=False)

    # MNIST DATASET

    if dataset_tag ==  DATASET_STACKEDMNIST:
        root = os.path.join(data_dir, 'mnist')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file) 
        load = args.load
        init_ds = Stacked_MNIST(root, load=load, source_root=root, 
                                train=True, num_channels = args.data.channels, 
                                imageSize=args.data.image_size,
                                device=args.device)

    if dataset_transfer_tag == DATASET_STACKEDMNIST:
        root = os.path.join(data_dir, 'mnist')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        final_ds = Stacked_MNIST(root, load=load, source_root=root,
                                train=True, num_channels = args.data.channels,
                                imageSize=args.data.image_size,
                                device=args.device)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3)

    # EMNIST DATASET

    if dataset_tag == DATASET_EMNIST:
        root = os.path.join(data_dir, 'EMNIST')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        init_ds = EMNIST(root, load=load, source_root=root,
                                train=True, num_channels = args.data.channels,
                                imageSize=args.data.image_size,
                                device=args.device)

    if dataset_transfer_tag == DATASET_EMNIST:
        root = os.path.join(data_dir, 'EMNIST')
        saved_file = os.path.join(root, "data.pt")
        load = os.path.exists(saved_file)
        load = args.load
        final_ds = EMNIST(root, load=load, source_root=root,
                                train=True, num_channels = args.data.channels,
                                imageSize=args.data.image_size,
                                device=args.device)
        mean_final = torch.tensor(0.)
        var_final = torch.tensor(1.*10**3)


    # FINAL (GAUSSIAN) DATASET (if no transfer)

    if not(args.transfer):
        if args.adaptive_mean:
            NAPPROX = 100
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean()
            mean_final = vec[0] * 0 + mean_final
            var_final = eval(args.var_final)
            final_ds = None
        elif args.final_adaptive:
            NAPPROX = 100
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var()
            final_ds = None
        else:
            mean_final = eval(args.mean_final)
            var_final = eval(args.var_final)
            final_ds = None
        

    return init_ds, final_ds, mean_final, var_final


def get_filtering_process(args):
    dataset_tag = getattr(args, DATASET)

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    if dataset_tag == DATASET_LORENZ:
        data_tag = args.data
        root = os.path.join(data_dir, 'lorenz')
        x, y = lorenz_process(root, data_tag)

    return x, y


def get_filtering_datasets(x_tm1, args):
    dataset_tag = getattr(args, DATASET)

    if dataset_tag == DATASET_LORENZ:
        data_tag = args.data
        init_ds = lorenz_ds(x_tm1, data_tag)

    assert not args.transfer

    if not(args.transfer):
        if args.adaptive_mean:
            NAPPROX = 100
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean()
            mean_final = vec[0] * 0 + mean_final
            var_final = eval(args.var_final)
            final_ds = None
        elif args.final_adaptive:
            NAPPROX = 100
            vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX)))[0]
            mean_final = vec.mean(axis=0)
            var_final = vec.var()
            final_ds = None
        else:
            mean_final = eval(args.mean_final)
            var_final = eval(args.var_final)
            final_ds = None

    return init_ds, final_ds, mean_final, var_final


# Logger
#--------------------------------------------------------------------------------

LOGGER = 'LOGGER'
LOGGER_PARAMS = 'LOGGER_PARAMS'

CSV_TAG = 'CSV'
NOLOG_TAG = 'NONE'

def get_logger(args, name):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {'directory': args.CSV_log_dir, 'name': name}
        return CSVLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()

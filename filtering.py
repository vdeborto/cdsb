import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFSequential
from bridge.runners.config_getters import get_filtering_datasets, get_filtering_process


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="filtering_config")
def main(args):
    # torch.set_default_dtype(torch.float64)

    print('Directory: ' + os.getcwd())

    x, y = get_filtering_process(args)
    T, xdim = x.shape

    x_ens = torch.randn([args.ens_size, xdim]) + 1

    x_ens_means = []
    x_ens_stds = []

    for t in range(T):
        with torch.no_grad():
            x_ens_repeat = x_ens.repeat(args.npar//args.ens_size, 1)
            init_ds_repeat, final_ds_repeat, mean_final, var_final = get_filtering_datasets(x_ens_repeat, args)

        ipf = IPFSequential(init_ds_repeat, final_ds_repeat, mean_final, var_final, args)
        if t == 0:
            print(ipf.net['b'])
            print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
        ipf.train()

        init_ds, _, _, _ = get_filtering_datasets(x_ens, args)
        init_x_batch, init_y_batch = init_ds.tensors
        x_ens = ipf.forward_backward_sample(init_x_batch, init_y_batch, y[t], n=args.n_ipf)[-1].detach().cpu()

        x_ens_means.append(x_ens.mean(0).numpy())
        x_ens_stds.append(x_ens.std(0).numpy())

        print(x[t].numpy())
        print(x_ens_means[-1])


if __name__ == '__main__':
    main()

import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFSequential, IPFAnalytic
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets, get_final_cond_model
from bridge.runners.accelerator import Accelerator


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="mnist")
def main(args):
    accelerator = Accelerator(train_batch_size=args.batch_size, cpu=args.device == 'cpu',
                              fp16=args.model.use_fp16, split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    final_cond_model = None
    if args.cond_final:
        final_cond_model = get_final_cond_model(args, init_ds)
    if args.Model in ['PolyCond', 'BasisCond', 'KRRCond']:
        ipf = IPFAnalytic(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                          final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)
    else:
        ipf = IPFSequential(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                            final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)
    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    ipf.train()


def hydra_argv_remapper(argv_map):
    """
    Call this function before main

    argv_map is a dict that remaps specific args to something else that hydra will gracefully not choke on

        ex: {'--foo':'standard.hydra.override.foo', '--bar':'example.bar'}
    """

    argv = []
    for v in sys.argv:
        if v[:2] == '--':
            argv = argv + v.split('=')
        else:
            argv.append(v)

    # Remap the args
    for k, v in argv_map.items():
        if k in argv:
            i = argv.index(k)
            if v is not None:
                new_arg = f"{v}={argv[i + 1]}"
                argv.append(new_arg)
            del argv[i:i + 2]

    # Replace sys.argv with our remapped argv
    sys.argv = argv


if __name__ == '__main__':
    hydra_argv_remapper({'--local_rank': None})
    main()  

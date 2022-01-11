import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFSequential
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets, get_final_cond_model

from accelerate import Accelerator


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="config")
def main(args):
    accelerator = Accelerator(fp16=False, cpu=args.device == 'cpu', split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())
    
    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    final_cond_model = None
    if args.cond_final:
        final_cond_model = get_final_cond_model(args, init_ds)

    ipf = IPFSequential(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                        final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)
    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    ipf.train()
    

if __name__ == '__main__':
    main()  

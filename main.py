import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFSequential
from bridge.runners.config_getters import get_datasets

from accelerate import Accelerator


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="config")
def main(args):
    accelerator = Accelerator(fp16=False, cpu=args.device == 'cpu', split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())
    
    init_ds, final_ds, mean_final, var_final = get_datasets(args)

    ipf = IPFSequential(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator)
    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    ipf.train()
    

if __name__ == '__main__':
    main()  

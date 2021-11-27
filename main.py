import torch
import hydra
import os,sys

sys.path.append('..')


from bridge.runners.ipf import IPFSequential
from bridge.runners.config_getters import get_datasets


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="config")
def main(args):

    print('Directory: ' + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args)

    ipf = IPFSequential(init_ds, final_ds, mean_final, var_final, args)
    ipf.train()
    

if __name__ == '__main__':
    main()  

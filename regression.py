import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFRegression
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from bridge.runners.accelerator import Accelerator


# SETTING PARAMETERS

def train(args):
    accelerator = Accelerator(train_batch_size=args.batch_size, cpu=args.device == 'cpu',
                              fp16=args.model.use_fp16, split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    assert not args.transfer
    init_ds, _, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    ipf = IPFRegression(init_ds, mean_final, var_final, args, accelerator=accelerator, valid_ds=valid_ds, test_ds=test_ds)

    accelerator.print(accelerator.state)
    accelerator.print(ipf.net)
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net.parameters() if p.requires_grad))
    ipf.train()

    return ipf.get_sample_net()


@hydra.main(config_path="./conf", config_name="mnist")
def main(args):
    train(args)


if __name__ == '__main__':
    main()

import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFSequential, IPFAnalytic
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets, get_final_cond_model
from bridge.runners.accelerator import Accelerator


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="test_mnist")
def main(args):
    accelerator = Accelerator(train_batch_size=args.batch_size, cpu=args.device == 'cpu',
                              fp16=args.model.use_fp16, split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    final_cond_model = None
    if args.cond_final:
        final_cond_model = get_final_cond_model(accelerator, args, init_ds)

    ipf = IPFSequential(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                        final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)
    ipf.checkpoint_iter = args.checkpoint_iter
    ipf.save_dls_dict = {"test": ipf.save_dls_dict["test"]}
    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    test_metrics = ipf.plot_and_test_step(ipf.checkpoint_iter, ipf.checkpoint_it, "b")

    accelerator.print("Valid: ", {k: test_metrics[k] for k in test_metrics.keys() if k[:6] == "valid/"})
    accelerator.print("Test: ", {k: test_metrics[k] for k in test_metrics.keys() if k[:5] == "test/"})
    accelerator.print("Cond: ", {k: test_metrics[k] for k in test_metrics.keys() if k[:5] == "cond/"})

if __name__ == '__main__':
    main()

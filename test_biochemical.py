import torch
import hydra
import os, sys

sys.path.append('..')

from bridge.runners.ipf import IPFSequential, IPFAnalytic
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets, get_final_cond_model
from bridge.runners.accelerator import Accelerator


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="test_config")
def main(args):
    accelerator = Accelerator(train_batch_size=args.batch_size, cpu=args.device == 'cpu',
                              fp16=args.model.use_fp16, split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    final_cond_model = None
    if args.cond_final:
        final_cond_model = get_final_cond_model(accelerator, args, init_ds)

    args.checkpoint_run = False
    ipf = IPFSequential(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                        final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds)

    accelerator.print(accelerator.state)
    accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))

    for n in range(1, 101):
        ipf.args.checkpoint_run = True

        ipf.checkpoint_it = n
        ipf.checkpoint_pass = "b"
        ipf.checkpoint_iter = args.checkpoint_iter

        ipf.args.sample_checkpoint_b = os.path.join(args.checkpoint_dir, f"sample_net_b_{n}_{args.checkpoint_iter}.ckpt")
        ipf.args.checkpoint_b = os.path.join(args.checkpoint_dir, f"net_b_{n}_{args.checkpoint_iter}.ckpt")
        # ipf.args.optimizer_checkpoint_b = os.path.join(args.checkpoint_dir, f"optimizer_b_{n}_{args.checkpoint_iter}.ckpt")

        if os.path.isfile(hydra.utils.to_absolute_path(os.path.join(args.checkpoint_dir, f"net_f_{n-1}_{args.checkpoint_iter}.ckpt"))):
            ipf.args.sample_checkpoint_f = os.path.join(args.checkpoint_dir, f"sample_net_f_{n-1}_{args.checkpoint_iter}.ckpt")
            ipf.args.checkpoint_f = os.path.join(args.checkpoint_dir, f"net_f_{n-1}_{args.checkpoint_iter}.ckpt")
            # ipf.args.optimizer_checkpoint_f = os.path.join(args.checkpoint_dir, f"optimizer_f_{n-1}_{args.checkpoint_iter}.ckpt")

        ipf.build_models()
        ipf.build_ema()


        test_metrics = ipf.plot_and_test_step(ipf.checkpoint_iter, n, "b")

        accelerator.print("Valid: ", {k: test_metrics[k] for k in test_metrics.keys() if k[:6] == "valid/"})
        accelerator.print("Test: ", {k: test_metrics[k] for k in test_metrics.keys() if k[:5] == "test/"})
        accelerator.print("Cond: ", {k: test_metrics[k] for k in test_metrics.keys() if k[:5] == "cond/"})

if __name__ == '__main__':
    main()

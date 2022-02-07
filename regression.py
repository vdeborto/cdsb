import os, sys
import numpy as np
import torch
import torch.nn.functional as functional
import hydra
from tqdm import tqdm

from bridge.runners.config_getters import get_datasets, get_valid_test_datasets, get_cond_model, get_optimizer, get_logger
from bridge.runners import repeater
from bridge.runners.accelerator import Accelerator

from torch.utils.data import DataLoader


# SETTING PARAMETERS

def train(args, final_cond_model=None):
    accelerator = Accelerator(fp16=False, cpu=args.device == 'cpu', split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    init_ds, _, _, _ = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)
    if final_cond_model is None:
        final_cond_model = get_cond_model(args)
    accelerator.print(accelerator.state)
    accelerator.print(final_cond_model)
    accelerator.print('Number of parameters:', sum(p.numel() for p in final_cond_model.parameters() if p.requires_grad))

    optimizer = get_optimizer(final_cond_model, args)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id + accelerator.process_index)

    kwargs = {"num_workers": args.num_workers,
              "pin_memory": args.pin_memory,
              "worker_init_fn": worker_init_fn,
              'drop_last': True}

    init_dl = DataLoader(init_ds, batch_size=args.batch_size, shuffle=True, **kwargs)

    # save_npar = max(args.plot_npar, args.test_npar)
    # test_batch_size = min(save_npar, args.test_batch_size)
    # valid_dl = DataLoader(valid_ds, batch_size=test_batch_size, **kwargs)
    # test_dl = DataLoader(test_ds, batch_size=test_batch_size, **kwargs)

    logger = get_logger(args, 'train_logs')

    accelerator.prepare(final_cond_model, init_dl, optimizer)
    init_dl = repeater(init_dl)

    for i in tqdm(range(1, args.cond_final_model.num_iter + 1)):
        batch_x, batch_y = next(init_dl)
        loss = functional.mse_loss(final_cond_model(batch_y), batch_x)
        accelerator.backward(loss)

        if args.grad_clipping:
            clipping_param = args.grad_clip
            total_norm = torch.nn.utils.clip_grad_norm_(final_cond_model.parameters(), clipping_param)
        else:
            total_norm = 0.

        optimizer.step()
        optimizer.zero_grad()

        if i == 1 or i % args.log_stride == 0 or i == args.cond_final_model.num_iter:
            logger.log_metrics({'loss': loss, 'grad_norm': total_norm}, step=i)
            print(final_cond_model(torch.tensor([0.18, 0.32, 0.42, 0.49, 0.54])))

        # if args.ema:
        #     ema_helper.update(final_cond_model)

    torch.save(accelerator.unwrap_model(final_cond_model).state_dict(), 'cond_final_model' + str(args.cond_final_model.num_iter) + '.ckpt')
    return final_cond_model


@hydra.main(config_path="./conf", config_name="config")
def main(args):
    train(args)


if __name__ == '__main__':
    main()

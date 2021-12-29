import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
import time


def CacheLoader(fb, sample_net, dataloader_b, dataloader_f, num_batches, langevin, ipf, n, device='cpu'):
    start = time.time()
    all_x = []
    all_y = []
    all_out = []
    all_steps = []

    sample_direction = 'f' if fb == 'b' else 'b'

    with torch.no_grad():
        for b in range(num_batches):
            batch_x, batch_y, _, mean_final, var_final = ipf.sample_batch(dataloader_b, dataloader_f, sample_direction)

            if (n == 1) & (fb == 'b'):
                x, y, out, steps_expanded = langevin.record_init_langevin(batch_x, batch_y,
                                                                          mean_final=mean_final,
                                                                          var_final=var_final)
            else:
                x, y, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch_x, batch_y)

            # store x, y, out, steps
            gather_x = ipf.accelerator.gather(x)
            gather_y = ipf.accelerator.gather(y)
            gather_out = ipf.accelerator.gather(out)
            gather_steps = ipf.accelerator.gather(steps_expanded)

            gather_x = gather_x.flatten(start_dim=0, end_dim=1).to(device)
            gather_y = gather_y.flatten(start_dim=0, end_dim=1).to(device)
            gather_out = gather_out.flatten(start_dim=0, end_dim=1).to(device)
            gather_steps = gather_steps.flatten(start_dim=0, end_dim=1).to(device)

            all_x.append(gather_x)
            all_y.append(gather_y)
            all_out.append(gather_out)
            all_steps.append(gather_steps)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_out = torch.cat(all_out, dim=0)
    all_steps = torch.cat(all_steps, dim=0)

    stop = time.time()
    ipf.accelerator.print('Cache size: {0}'.format(all_x.shape))
    ipf.accelerator.print("Load time: {0}".format(stop-start))
    ipf.accelerator.print("Out mean: {0}".format(all_out.mean().item()))
    ipf.accelerator.print("Out std: {0}".format(all_out.std().item()))

    return TensorDataset(all_x, all_y, all_out, all_steps)

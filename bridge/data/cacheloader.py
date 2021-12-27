import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
import time


def CacheLoader(fb, sample_net, dataloader_b, dataloader_f, num_batches, langevin, ipf, n, device='cpu'):
    start = time.time()
    data_x = []
    data_y = []
    data_out = []
    data_steps = []

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
            all_x = ipf.accelerator.gather(x)
            all_y = ipf.accelerator.gather(y)
            all_out = ipf.accelerator.gather(out)
            all_steps = ipf.accelerator.gather(steps_expanded)

            all_x = all_x.flatten(start_dim=0, end_dim=1).to(device)
            all_y = all_y.flatten(start_dim=0, end_dim=1).to(device)
            all_out = all_out.flatten(start_dim=0, end_dim=1).to(device)
            all_steps = all_steps.flatten(start_dim=0, end_dim=1).to(device)

            data_x.append(all_x)
            data_y.append(all_y)
            data_out.append(all_out)
            data_steps.append(all_steps)

    data_x = torch.cat(data_x, dim=0)
    data_y = torch.cat(data_y, dim=0)
    data_out = torch.cat(data_out, dim=0)
    data_steps = torch.cat(data_steps, dim=0)

    stop = time.time()
    ipf.accelerator.print('Cache size: {0}'.format(data_x.shape))
    ipf.accelerator.print("Load time: {0}".format(stop-start))

    return TensorDataset(data_x, data_y, data_out, data_steps)

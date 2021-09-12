import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time

class CacheLoader(Dataset):
    def __init__(self, fb, 
                 sample_net, 
                 dataloader_b, 
                 num_batches, 
                 langevin, 
                 n,  
                 mean, std, 
                 batch_size, device='cpu', 
                 dataloader_f=None,
                 transfer=False): 

        super().__init__()
        start = time.time()
        shape = langevin.d
        num_steps = langevin.num_steps
        self.data = torch.zeros((num_batches, batch_size*num_steps, 3, *shape)).to(device)#.cpu()
        self.steps_data = torch.zeros((num_batches, batch_size*num_steps,1), dtype=torch.long).to(device)#.cpu() # steps
        with torch.no_grad():
            for b in range(num_batches):
                if fb=='b':
                    batch = next(dataloader_b)
                    batch_x = batch[0]
                    batch_y = batch[1]
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                elif fb =='f' and transfer:
                    batch = next(dataloader_f)
                    batch_x = batch[0]
                    batch_y = batch[1]
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)                    
                else:                    
                    batch_x = mean + std*torch.randn((batch_size, *shape), device=device)
                    batch_y = next(dataloader_b)[1]                    
                    batch_y = batch_y.to(device)                    
                
                if (n == 1) & (fb=='b'):
                    x, y, out, steps_expanded = langevin.record_init_langevin(batch_x, batch_y)
                else:
                    x, y, out, steps_expanded = langevin.record_langevin_seq(sample_net, batch_x, batch_y, ipf_it=n)
                
                # store x, out
                x = x.unsqueeze(2)
                y = y.unsqueeze(2)
                out = out.unsqueeze(2)
                batch_data = torch.cat((x, y, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data 
                
                # store steps
                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps 
        
        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)
        
        stop = time.time()
        # print('Cache size: {0}'.format(self.data.shape))
        # print("Load time: {0}".format(stop-start))
    
    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        y = item[1]
        out = item[2]
        steps = self.steps_data[index]
        return x, y, out, steps

    def __len__(self):
        return self.data.shape[0]
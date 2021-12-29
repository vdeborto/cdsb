import os, shutil
import urllib
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import save_image


class Stacked_MNIST(Dataset):
    def __init__(self, args, root="./dataset", load=True, source_root=None, imageSize=28,
                 train=True, num_channels=3, device='cpu'):  # load=True means loading the dataset from existed files.
        super(Stacked_MNIST, self).__init__()
        self.num_channels = min(3,num_channels)
        if load:
            self.data = torch.load(os.path.join(root, "data.pt"))
            self.targets = torch.load(os.path.join(root, "targets.pt"))
        else:
            if source_root is None:
                source_root = "./datasets"

            source_data = torchvision.datasets.MNIST(source_root, train=train, transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]), download=True)
            self.data = torch.zeros((0, self.num_channels, imageSize, imageSize))
            self.targets = torch.zeros((0), dtype=torch.int64)
            # has 60000 images in total
            dataloader_R = DataLoader(source_data, batch_size=100, shuffle=True)
            dataloader_G = DataLoader(source_data, batch_size=100, shuffle=True)
            dataloader_B = DataLoader(source_data, batch_size=100, shuffle=True)

            im_dir = root + '/im'
            if os.path.exists(im_dir):
                shutil.rmtree(im_dir)
            os.makedirs(im_dir)

            for (xR, yR), (xG, yG), (xB, yB) in tqdm(zip(dataloader_R, dataloader_G, dataloader_B)):
                x = torch.cat([xR, xG, xB][-self.num_channels:], dim=1)
                y = (100 * yR + 10 * yG + yB) % 10**self.num_channels
                self.data = torch.cat((self.data, x), dim=0)
                self.targets = torch.cat((self.targets, y), dim=0)
                    
            if not os.path.isdir(root):
                os.makedirs(root)
            torch.save(self.data, os.path.join(root, "data.pt"))
            torch.save(self.targets, os.path.join(root, "targets.pt"))
            save_image(x, os.path.join(root, "data_x.png"), nrow=10)

            for k in range(args.test_npar):
                im = self.data[k]
                filename = root + '/im/{:05}.jpg'.format(k)
                save_image(im, filename)

        self.data = self.data#.to(device)
        self.targets = self.targets#.to(device)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        return img, targets

    def __len__(self):
        return len(self.targets)


class Cond_Stacked_MNIST(Stacked_MNIST):
    def __init__(self, args, root="./dataset", load=True, source_root=None, imageSize=28,
                 train=True, num_channels=3, device='cpu'):
        super().__init__(args, root, load, source_root, imageSize, train, num_channels, device)
        if args.task == 'SuperRes':
            factor = args.factor
            downsample_kernel = torch.ones(num_channels, 1, factor, factor)
            downsample_kernel = downsample_kernel / factor ** 2

            self.data_y = torch.nn.functional.conv2d(self.data, downsample_kernel, stride=factor,
                                                     groups=num_channels)
            self.data_y = torch.nn.functional.interpolate(self.data_y, (imageSize, imageSize))

            save_image(self.data_y[-100:], os.path.join(root, "data_y.png"), nrow=10)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        img, targets = self.data[index], self.data_y[index]

        return img, targets

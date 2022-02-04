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
    def __init__(self, root="./data/mnist/", source_root="./data/mnist/", imageSize=28, train=True, num_channels=3):
        super().__init__()
        torch.manual_seed(0)

        self.num_channels = min(3, num_channels)

        source_data = torchvision.datasets.MNIST(source_root, train=train, transform=transforms.Compose([
            transforms.Resize(imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]), download=True)
        self.data = torch.zeros((0, self.num_channels, imageSize, imageSize))
        self.targets = torch.zeros((0), dtype=torch.int64)

        dataloader_R = DataLoader(source_data, batch_size=100, shuffle=True)
        dataloader_G = DataLoader(source_data, batch_size=100, shuffle=True)
        dataloader_B = DataLoader(source_data, batch_size=100, shuffle=True)

        for (xR, yR), (xG, yG), (xB, yB) in tqdm(zip(dataloader_R, dataloader_G, dataloader_B)):
            x = torch.cat([xR, xG, xB][-self.num_channels:], dim=1)
            y = (100 * yR + 10 * yG + yB) % 10**self.num_channels
            self.data = torch.cat((self.data, x), dim=0)
            self.targets = torch.cat((self.targets, y), dim=0)

        self.root = root
        if not os.path.isdir(root):
            os.makedirs(root)

        if train:
            for split in ['train', 'valid']:
                self.save_split_data(split)
        else:
            split = 'test'
            self.save_split_data(split)

    def save_split_data(self, split):
        if split == 'train':
            data = self.data[:-10000]
            targets = self.targets[:-10000]

            assert torch.all(targets[:10] == torch.tensor([1, 1, 4, 4, 9, 6, 7, 5, 3, 4]))
        elif split == 'valid':
            data = self.data[-10000:]
            targets = self.targets[-10000:]

            assert torch.all(targets[:10] == torch.tensor([3, 4, 4, 4, 3, 4, 1, 1, 2, 9]))
        elif split == 'test':
            data = self.data
            targets = self.targets

            assert torch.all(targets[:10] == torch.tensor([5, 1, 7, 1, 6, 1, 7, 5, 0, 2]))

        torch.save(data, os.path.join(self.root, f"data_x_{split}.pt"))
        torch.save(targets, os.path.join(self.root, f"targets_{split}.pt"))

        save_image(data[:100], os.path.join(self.root, f"data_x_{split}.png"), nrow=10)

        im_dir = self.root + f'/im_{split}'
        if os.path.exists(im_dir):
            shutil.rmtree(im_dir)
        os.makedirs(im_dir)

        for k in range(len(data)):
            im = data[k]
            filename = os.path.join(im_dir, '{:05}.png'.format(k))
            save_image(im, filename)
        

class Cond_Stacked_MNIST(Dataset):
    def __init__(self, data_tag, root="./data/mnist/", load=True, split='train', num_channels=3):
        super().__init__()
        torch.manual_seed(0)

        self.data_x = torch.load(os.path.join(root, f"data_x_{split}.pt"))
        self.targets = torch.load(os.path.join(root, f"targets_{split}.pt"))
        imageSize = self.data_x.shape[2]
        assert num_channels == self.data_x.shape[1]
        assert imageSize == 28

        if load:
            self.data_y = torch.load(os.path.join(root, f"data_y_{data_tag}_{split}.pt"))
        else:
            task = data_tag.split("_")
            if task[0] == 'superres':
                factor = int(task[1])
                downsample_kernel = torch.ones(num_channels, 1, factor, factor)
                downsample_kernel = downsample_kernel / factor ** 2

                self.data_y = torch.nn.functional.conv2d(self.data_x, downsample_kernel, stride=factor,
                                                         groups=num_channels)
                self.data_y = torch.nn.functional.interpolate(self.data_y, (imageSize, imageSize))
            elif task[0] == 'inpaint':
                mask = torch.zeros([1, imageSize, imageSize])
                if task[1] == 'center':
                    mask[:, imageSize//4:-imageSize//4, imageSize//4:-imageSize//4] = 1
                self.data_y = self.data_x * (1 - mask)
            else:
                raise NotImplementedError

            torch.save(self.data_y, os.path.join(root, f"data_y_{data_tag}_{split}.pt"))
            save_image(self.data_y[:100], os.path.join(root, f"data_y_{data_tag}_{split}.png"), nrow=10)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        img, targets = self.data_x[index], self.data_y[index]

        return img, targets

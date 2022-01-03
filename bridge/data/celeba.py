import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class Cond_CelebA(CelebA):
    def __init__(self, data_tag, root, split="train", target_type=[], transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, target_type=target_type, transform=transform, target_transform=target_transform, download=download)

        self.data_tag = data_tag

    def __getitem__(self, item):
        X, _ = super().__getitem__(item)
        imageSize = X.shape[-1]

        task = self.data_tag.split("_")
        if task[0] == 'superres':
            factor = int(task[1])

            Y = torch.nn.functional.interpolate(X.unsqueeze(0), (imageSize//factor, imageSize//factor), mode='area')
            Y = torch.nn.functional.interpolate(Y, (imageSize, imageSize)).squeeze(0)

        else:
            raise NotImplementedError

        return X, Y



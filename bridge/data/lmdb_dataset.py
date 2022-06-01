import os
import io
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import lmdb
from PIL import Image

from .utils import to_uint8_tensor
from gitmodules.matlab_imresize.imresize import imresize


def num_samples(dataset, split):
    if dataset == 'celeba':
        if split == "train":
            return 162770
        elif split == "validation":
            return 19867
        elif split == "test":
            return 19962
    elif dataset == 'celebahq':
        if split == "train":
            return 27000
        elif split == "validation":
            return 3000
    elif dataset == 'ffhq':
        if split == "train":
            return 63000
        elif split == "validation":
            return 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(Dataset):
    def __init__(self, root, name='', split="train", transform=None, is_encoded=False):
        self.split = split
        self.name = name
        self.transform = transform
        lmdb_path = os.path.join(root, self.split + '.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

        if not self.is_encoded:
            with self.data_lmdb.begin(write=False, buffers=True) as txn:
                data = txn.get(str(0).encode())
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                self.size = int(np.sqrt(len(img) / 3))

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8).reshape((self.size, self.size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return num_samples(self.name, self.split)


class Cond_LMDBDataset(LMDBDataset):
    def __init__(self, data_tag, root, name, split="train", transform=None, is_encoded=False):
        super().__init__(root, name=name, split=split, transform=transform, is_encoded=is_encoded)

        self.data_tag = data_tag

    def __getitem__(self, item):
        X, _ = super().__getitem__(item)
        imageSize = X.shape[-1]

        task = self.data_tag.split("_")
        if task[0] == 'superres':
            factor = int(task[1])

            Y = torch.nn.functional.interpolate(X.unsqueeze(0), (imageSize // factor, imageSize // factor), mode='area')

            if len(task) > 2:
                if task[2] == 'noise':
                    std = float(task[3])
                    Y = Y + torch.randn_like(Y) * std

            Y = torch.nn.functional.interpolate(Y, (imageSize, imageSize)).squeeze(0)

        elif task[0] == 'inpaint':
            mask = torch.zeros([1, imageSize, imageSize])
            if task[1] == 'center':
                mask[:, imageSize//4:-imageSize//4, imageSize//4:-imageSize//4] = 1
            elif task[1] == 'left':
                mask[:, :, :imageSize//2] = 1
            elif task[1] == 'right':
                mask[:, :, imageSize//2:] = 1
            Y = X * (1 - mask)

        else:
            raise NotImplementedError

        return X, Y


class Cond_CelebA160(LMDBDataset):
    def __init__(self, data_tag, root, name, split="train", transform=None, is_encoded=False):
        super().__init__(root, name=name, split=split, transform=transform, is_encoded=is_encoded)
        assert self.size == 160
        self.data_tag = data_tag
        self.np_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, item):
        X, _ = super().__getitem__(item)
        imageSize = X.shape[-1]
        X_np = to_uint8_tensor(X).numpy().transpose((1, 2, 0))

        task = self.data_tag.split("_")
        if task[0] == 'superres':
            factor = int(task[1])
            Y_np = imresize(X_np, output_shape=(imageSize // factor, imageSize // factor))
            Y = self.np_transform(Y_np).unsqueeze(0)
            Y = torch.nn.functional.interpolate(Y, (imageSize, imageSize)).squeeze(0)

        else:
            raise NotImplementedError

        return X, Y

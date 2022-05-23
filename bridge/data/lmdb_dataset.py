import os
import io
import numpy as np
import torch
from torch.utils.data import Dataset
import lmdb
from PIL import Image


def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'celeba64':
        return 162770 if train else 19867
    elif dataset == 'imagenet-oord':
        return 1281147 if train else 50000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
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
        return num_samples(self.name, self.train)


class Cond_LMDBDataset(LMDBDataset):
    def __init__(self, data_tag, root, name, train=True, transform=None):
        super().__init__(root, name=name, train=train, transform=transform)

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

        else:
            raise NotImplementedError

        return X, Y

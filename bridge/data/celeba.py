import torch
from torchvision.datasets import CelebA as _CelebA


class CelebA(_CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, target_type=target_type, transform=transform, target_transform=target_transform, download=download)
        self.filename = np.array(self.filename).astype(np.string_)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", str(self.filename[index], encoding='utf-8')))
        
        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target


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

            if len(task) > 2:
                if task[2] == 'noise':
                    std = float(task[3])
                    Y = Y + torch.randn_like(Y) * std

            Y = torch.nn.functional.interpolate(Y, (imageSize, imageSize)).squeeze(0)

        else:
            raise NotImplementedError

        return X, Y



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from gitmodules.pulse.PULSE import PULSE
from bridge.data.utils import normalize_tensor, unnormalize_tensor


class PULSEModel(nn.Module):
    def __init__(self, hr_size, lr_size):
        super().__init__()
        self.model = PULSE(cache_dir=hydra.utils.to_absolute_path("gitmodules/pulse/cache"))
        self.hr_size = hr_size
        self.lr_size = lr_size

        self.kwargs = {
            "seed": None,
            "loss_str": "100*L2+0.05*GEOCROSS",
            "eps": 2e-3,
            "noise_type": "trainable",
            "num_trainable_noise_layers": 5,
            "tile_latent": False,
            "bad_noise_layers": "17",
            "opt_name": 'adam',
            "learning_rate": 0.4,
            "steps": 100,
            "lr_schedule": 'linear1cycledrop',
            "batch_size": 16,
            "save_intermediate": False
        }

    def forward(self, x):
        x = normalize_tensor(x)
        x = torch.nn.functional.interpolate(x, (self.lr_size, self.lr_size))
        dataloader = DataLoader(x, batch_size=self.kwargs["batch_size"])
        out = []

        for batch in dataloader:
            (HR, _), = self.model(batch, **self.kwargs)
            out.append(HR.detach().clamp(0, 1))
        out = torch.cat(out, dim=0)
        out = torch.nn.functional.interpolate(out, (self.hr_size, self.hr_size))
        out = unnormalize_tensor(out)
        return out

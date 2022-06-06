import torch
import torch.nn as nn
from gitmodules.SRFlow.code.test import load_model
from bridge.data.utils import normalize_tensor, unnormalize_tensor


class SRFlowModel(nn.Module):
    def __init__(self, hr_size, lr_size, conf_path, device, temperature=0.8):
        super().__init__()
        self.model, _ = load_model(conf_path)
        self.hr_size = hr_size
        self.lr_size = lr_size
        assert hr_size == 160
        self.device = device
        self.model.to(device)
        self.temperature = temperature

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, (self.lr_size, self.lr_size))
        x = normalize_tensor(x)
        while True:
            out = self.model.get_sr(lq=x, heat=self.temperature)
            out = unnormalize_tensor(out)
            if not torch.any(torch.isnan(out)):
                break
            print("SRFlowModel output nan, retrying")
        return out

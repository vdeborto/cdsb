import torch
import torch.nn as nn
from gitmodules.SRFlow.code.test import load_model
from bridge.data.utils import normalize_tensor, unnormalize_tensor


class SRFlowModel(nn.Module):
    def __init__(self, conf_path, temperature=0.8):
        super().__init__()
        self.model, _ = load_model(conf_path)
        self.temperature = temperature

    def forward(self, x):
        x = normalize_tensor(x)
        out = self.model.get_sr(lq=x, heat=self.temperature)
        out = unnormalize_tensor(out)
        return out

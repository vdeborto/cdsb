import torch
import torchvision.utils as vutils


def save_image(tensor, fp, format=None, **kwargs):
    tensor = tensor / 2 + 0.5
    vutils.save_image(tensor, fp, format=format, **kwargs)

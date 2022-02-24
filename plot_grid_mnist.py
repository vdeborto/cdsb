import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.utils as vutils
from bridge.data.stackedmnist import Cond_Stacked_MNIST
import torchvision.transforms as transforms

from bridge.data.metrics import PSNR, SSIM, FID


DPI = 300


def to_uint8_tensor(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

np.random.seed(1234)

# base_dir = "experiments/stackedmnist_superres_4/2022-02-10/cfg-model.dropout=0.1,n_ipf=1,num_cache_batches=8,num_iter=500000,num_steps=5/22-52-21/im/version_0/470000_b_1/test"
base_dir = "experiments/stackedmnist_superres_4/2022-02-12/cfg-cond_final=True,final_adaptive=True,gamma_max=0.05,model.dropout=0.1,n_ipf=5,num_cache_batches=8,num_iter=100000,num_steps=5/17-11-00/im/version_0/100000_b_3/test"
im_dir = os.path.join(base_dir, "im")
num_im = 64
nrow = 8
im_idx_list = np.random.choice(np.arange(10000), num_im, replace=False)
im_name_list = [f'{i:05}' + ".png" for i in im_idx_list]
add_title = True
n = base_dir.split("_")[-1].split("/")[0]


ds = Cond_Stacked_MNIST("superres_4", "data/mnist", load=True, split='test', num_channels=1)

im_list = []
for im_name in im_name_list:
    im_list.append(torchvision.io.read_image(os.path.join(im_dir, im_name), torchvision.io.ImageReadMode(1)).float()/255)


data_x_im_list = []
data_y_im_list = []
for i in im_idx_list:
    x, y = ds[i]
    data_x_im_list.append(x/2+0.5)
    data_y_im_list.append(y/2+0.5)

if add_title:
    im_list = torch.stack(im_list, 0)
    data_x_im_list = torch.stack(data_x_im_list, 0)
    data_y_im_list = torch.stack(data_y_im_list, 0)

    uint8_x_init = to_uint8_tensor(data_x_im_list)

    def save_image_with_metrics(batch_x, filename, split, **kwargs):
        plt.clf()

        plt.figure(figsize=(3.9,3.9))

        uint8_batch_x = to_uint8_tensor(batch_x)

        psnr = PSNR(data_range=255.)
        psnr_result = psnr(uint8_batch_x, uint8_x_init).item()
        psnr.reset()

        ssim = SSIM(data_range=255.)
        ssim_result = ssim(uint8_batch_x, uint8_x_init).item()
        ssim.reset()

        uint8_batch_x_grid = vutils.make_grid(uint8_batch_x, **kwargs).permute(1, 2, 0)
        plt.imshow(uint8_batch_x_grid)

        if split == "last":
            plt.title('CDSB iteration:' + str(n) + '  psnr:' + str(round(psnr_result, 2)) + '  ssim:' + str(round(ssim_result, 2)))
        elif split == "data_y":
            plt.title('psnr:' + str(round(psnr_result, 2)) + '  ssim:' + str(round(ssim_result, 2)))
        else:
            plt.title(' ')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', dpi=DPI)
        plt.close()

    save_image_with_metrics(im_list, os.path.join(base_dir, "im_grid_last_new.png"), "last", nrow=nrow)
    save_image_with_metrics(data_x_im_list, os.path.join(base_dir, "im_grid_data_x_new.png"), "data_x", nrow=nrow)
    save_image_with_metrics(data_y_im_list, os.path.join(base_dir, "im_grid_data_y_new.png"), "data_y", nrow=nrow)

else:
    vutils.save_image(im_list, os.path.join(base_dir, "im_grid_last_new.png"), nrow=nrow)
    vutils.save_image(data_x_im_list, os.path.join(base_dir, "im_grid_data_x_new.png"), nrow=nrow)
    vutils.save_image(data_y_im_list, os.path.join(base_dir, "im_grid_data_y_new.png"), nrow=nrow)
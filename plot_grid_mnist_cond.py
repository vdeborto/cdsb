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

base_dir = "experiments/stackedmnist_inpaint_center/2022-02-15/cfg-model.dropout=0.1,n_ipf=5,num_cache_batches=4,num_iter=100000,num_steps=10/18-02-54/im/version_0/100000_b_5/cond"
fwdbwd = False
im_dir = os.path.join(base_dir, "im_fwdbwd" if fwdbwd else "im_")
num_im = 100
nrow = 8
im_idx_list = np.arange(nrow)
im_name_list = [f'{i:05}' + ".png" for i in np.arange(num_im)]
add_title = True
n = base_dir.split("_")[-1].split("/")[0]


ds = Cond_Stacked_MNIST("superres_4", "data/mnist", load=True, split='test', num_channels=1)



batch_x = []
x_init_cond = []
y_cond = []
for i in im_idx_list:
    x, y = ds[i]
    x_init_cond.append(x / 2 + 0.5)
    y_cond.append(y / 2 + 0.5)
    batch_x_c = []
    for im_name in im_name_list:
        batch_x_c.append(torchvision.io.read_image(os.path.join(im_dir, str(i), im_name), torchvision.io.ImageReadMode(1)).float() / 255)
    batch_x.append(torch.stack(batch_x_c, 0))

if add_title:
    batch_x = torch.stack(batch_x, 0)
    x_init_cond = torch.stack(x_init_cond, 0)
    y_cond = torch.stack(y_cond, 0)

    uint8_x_init = to_uint8_tensor(x_init_cond)


    def save_image_with_metrics(batch_x, filename):
        # batch_x shape (y_cond, b, c, h, w)
        plt.clf()
        ncol = 10 if x_init_cond is not None else 9
        plt.figure(figsize=(ncol, batch_x.shape[0]))

        uint8_batch_x = to_uint8_tensor(batch_x)
        batch_x_mean = batch_x.mean(1)
        batch_x_std = batch_x.std(1)
        uint8_batch_x_mean = to_uint8_tensor(batch_x_mean)

        plt_idx = 1

        def subplot_imshow(tensor, plt_idx):
            ax = plt.subplot(len(y_cond), ncol, plt_idx)
            ax.axis('off')
            ax.imshow(tensor.permute(1, 2, 0))


        for j in range(len(y_cond)):
            if x_init_cond is not None:
                subplot_imshow(uint8_x_init[j].expand(3, -1, -1), plt_idx)
                plt_idx += 1
            subplot_imshow(to_uint8_tensor(y_cond[j]).expand(3, -1, -1), plt_idx)
            plt_idx += 1
            for k in range(6):
                subplot_imshow(uint8_batch_x[j, k].expand(3, -1, -1), plt_idx)
                plt_idx += 1
            if batch_x_mean.shape[1] == 1:
                subplot_imshow(batch_x_mean[j], plt_idx)
            elif batch_x_mean.shape[1] == 3:
                subplot_imshow(uint8_batch_x_mean[j], plt_idx)
            else:
                raise ValueError
            plt_idx += 1
            subplot_imshow(batch_x_std[j], plt_idx)
            plt_idx += 1

        psnr = PSNR(data_range=255.)
        psnr_result = psnr(uint8_batch_x.flatten(end_dim=1),
                           uint8_x_init.unsqueeze(1).expand(-1, batch_x.shape[1], -1, -1, -1).flatten(end_dim=1)).item()
        psnr.reset()
        mean_psnr_result = psnr(uint8_batch_x_mean, uint8_x_init).item()
        psnr.reset()

        ssim = SSIM(data_range=255.)
        ssim_result = ssim(uint8_batch_x.flatten(end_dim=1),
                           uint8_x_init.unsqueeze(1).expand(-1, batch_x.shape[1], -1, -1, -1).flatten(end_dim=1)).item()
        ssim.reset()
        mean_ssim_result = ssim(uint8_batch_x_mean, uint8_x_init).item()
        ssim.reset()
        if add_title:

            plt.suptitle('CDSB iteration: ' + str(n) +
                        '\npsnr: ' + str(round(psnr_result, 2)) + '   mean psnr: ' + str(round(mean_psnr_result, 2)) +
                        '\nssim: ' + str(round(ssim_result, 2)) + '   mean ssim: ' + str(round(mean_ssim_result, 2)), fontsize=16)
        plt.savefig(filename, bbox_inches='tight', dpi=DPI)
        plt.close()

    save_image_with_metrics(batch_x, os.path.join(base_dir, "cond_im_grid_fwdbwd_last_new.png" if fwdbwd else "cond_im_grid__last_new.png"))

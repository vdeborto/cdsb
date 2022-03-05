import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.utils as vutils
from bridge.data.celeba import Cond_CelebA
import torchvision.transforms as transforms

from bridge.data.metrics import PSNR, SSIM, FID


DPI = 100


def to_uint8_tensor(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

np.random.seed(1234)

base_dir = "experiments/celeba_superres_4_noise_0.2/test/cfg-n_ipf=1,num_cache_batches=35,num_iter=500000,num_steps=20,plot_npar=20000/10-43-57/im/version_0/10000_b_1/test"
im_dir = os.path.join(base_dir, "im")
repeat = 5
num_im = 25
nrow = 5
im_idx_list_repeat = np.random.choice(np.arange(19800), num_im*repeat, replace=False).reshape((repeat, num_im))
im_name_list_repeat = [[f'{i:05}' + ".png" for i in l] for l in im_idx_list_repeat]
add_title = False
n = base_dir.split("_")[-1].split("/")[0]

os.makedirs(os.path.join(base_dir, "sample_grid"), exist_ok=True)

test_transform = [transforms.CenterCrop(140), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
ds = Cond_CelebA("superres_4_noise_0.2", "data", split="test", transform=transforms.Compose(test_transform))

for r in range(repeat):
    im_idx_list = im_idx_list_repeat[r]
    im_name_list = im_name_list_repeat[r]

    im_list = []
    for im_name in im_name_list:
        im_list.append(torchvision.io.read_image(os.path.join(im_dir, im_name), torchvision.io.ImageReadMode(3)).float()/255)


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

            plt.figure(figsize=(4, 4))

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

        save_image_with_metrics(im_list, os.path.join(base_dir, "sample_grid", f"im_grid_last_repeat{r}.png"), "last", nrow=nrow)
        save_image_with_metrics(data_x_im_list, os.path.join(base_dir, "sample_grid", f"im_grid_data_x_repeat{r}.png"), "data_x", nrow=nrow)
        save_image_with_metrics(data_y_im_list, os.path.join(base_dir, "sample_grid", f"im_grid_data_y_repeat{r}.png"), "data_y", nrow=nrow)

    else:
        vutils.save_image(im_list, os.path.join(base_dir, "sample_grid", f"im_grid_last_repeat{r}.png"), nrow=nrow)
        vutils.save_image(data_x_im_list, os.path.join(base_dir, "sample_grid", f"im_grid_data_x_repeat{r}.png"), nrow=nrow)
        vutils.save_image(data_y_im_list, os.path.join(base_dir, "sample_grid", f"im_grid_data_y_repeat{r}.png"), nrow=nrow)
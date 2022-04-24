import os
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

def save_compared_images(opt, generator, encoder, dataloader, device):
    generator.load_state_dict(torch.load("results/generator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    encoder.to(device).eval()

    os.makedirs("results/images_diff", exist_ok=True)
    os.makedirs("results/images_diff/individuals", exist_ok=True)
    for i, (img, label) in enumerate(dataloader):
        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)

        compared_images = torch.empty(real_img.shape[0] * 3,
                                      *real_img.shape[1:])
        #print(compared_images.shape)
        compared_images[0::3] = real_img
        compared_images[1::3] = fake_img
        compared_images[2::3] = real_img - fake_img
        img_diff = ((real_img - fake_img)**2).detach().cpu().numpy()
        img_diff = np.sum(img_diff, axis=1, keepdims=False)

        save_image(compared_images.data,
                   f"results/images_diff/{opt.n_grid_lines * (i + 1):06}.png",
                   nrow=3, normalize=True)
        #real_img_np = real_img.permute(0,2,3,1).detach().cpu().numpy()
        #fake_img_np = fake_img.permute(0, 2, 3, 1).detach().cpu().numpy()
        for j in range(real_img.shape[0]):
            plt.imshow(img_diff[j], cmap='viridis')
            plt.axis('off')
            plt.savefig(f"results/images_diff/individuals/{opt.n_grid_lines * (i) +j:06}_diff.png", bbox_inches='tight')
            save_image(real_img[j].data,
                       f"results/images_diff/individuals/{opt.n_grid_lines * (i) +j:06}_real.png",
                        normalize=True)
            save_image(fake_img[j].data,
                       f"results/images_diff/individuals/{opt.n_grid_lines * (i) +j:06}_fake.png",
                       normalize=True)

        #np.savez(f"results/images_diff/{opt.n_grid_lines * (i + 1):06}_real.npz",
        #         img=real_img.detach().cpu().numpy())
        #np.savez(f"results/images_diff/{opt.n_grid_lines * (i + 1):06}_fake.npz",
        #         img=fake_img.detach().cpu().numpy())
        #np.savez(f"results/images_diff/{opt.n_grid_lines * (i + 1):06}.npz",
        #         img_diff=img_diff.detach().cpu().numpy())

        if opt.n_iters is not None and opt.n_iters == i:
            break

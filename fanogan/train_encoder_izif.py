import os
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional
from torchvision.utils import save_image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

"""
These codes are:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


def train_encoder_izif(opt, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=opt.lr, betas=(opt.b1, opt.b2))

    os.makedirs("results/images_e", exist_ok=True)

    padding_epoch = len(str(opt.n_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)
            do_noise = random.random()
            do_inp = random.random()
            if opt.encoder_denoise_level>0.0 and do_noise>0.5:
                in_imgs = real_imgs + torch.rand(real_imgs.size(), dtype=real_imgs.dtype, device=real_imgs.device)*opt.encoder_denoise_level
            else:
                in_imgs = real_imgs
            if opt.encoder_inpainting and do_inp>0.5:
                in_imgs = torchvision.transforms.functional.rgb_to_grayscale(in_imgs, num_output_channels=3)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(in_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            if opt.aux_recon:
                real_features = discriminator.forward_features(in_imgs)
            else:
                real_features = discriminator.forward_features(real_imgs)
            # Fake features
            fake_features = discriminator.forward_features(fake_imgs)

            # izif architecture
            if opt.aux_recon:
                loss_imgs = criterion(fake_imgs, in_imgs)
            else:
                loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features
            if opt.use_ssim > 0.0:
                e_loss+= opt.use_ssim * ( 1 - ssim( real_imgs, fake_imgs, data_range=1.0, size_average=True))

            e_loss.backward()
            optimizer_E.step()

            # Output training log every n_critic steps
            if i % opt.n_critic == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{opt.n_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

                if batches_done % opt.sample_interval == 0:
                    fake_z = encoder(fake_imgs)
                    reconfiguration_imgs = generator(fake_z)
                    save_image(reconfiguration_imgs.data[:25],
                               f"results/images_e/{batches_done:06}.png",
                               nrow=5, normalize=True)

                batches_done += opt.n_critic
    torch.save(encoder.state_dict(), "results/encoder")

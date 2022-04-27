import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision4ad.datasets import MVTecAD
import numpy as np
# from fanogan.test_anomaly_detection import test_anomaly_detection
from mvtecad_pytorch.dataset import MVTecADDataset
from model import Generator, Discriminator, Encoder
from sklearn.metrics import roc_auc_score
from torch.utils.model_zoo import tqdm
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def cal_pro_scores(masks, amaps):
    loc_np = np.asarray(amaps).flatten()

    mask_np = np.asarray(masks).flatten()
    mask_np = np.where(mask_np > 0.5, 1, 0)
    print(loc_np.shape, loc_np.dtype)
    PRO_score = roc_auc_score(mask_np, loc_np, max_fpr=0.3)
    print("PRO Score: ", PRO_score)


def test_anomaly_detection(opt, generator, discriminator, encoder,
                           dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss()

    with open("results/score.csv", "w") as f:
        f.write("label,img_distance,anomaly_score,z_distance\n")
    masks = []
    amaps = []
    for (img, mask, label) in tqdm(dataloader):

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)
        if (opt.gaussian_blur):
            blurrer = transforms.GaussianBlur((7, 7), (opt.gaussian_blur_sigma1, opt.gaussian_blur_sigma2))
            fake_img = blurrer(fake_img)
            real_img = blurrer(real_img)
        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)
        pix_scores = torch.sum((real_img - fake_img) ** 2, 1)
        # pix_scores -= pix_scores.min()
        # pix_scores /= pix_scores.max()
        masks.append(mask.unsqueeze(0).detach().cpu().numpy())
        amaps.append(pix_scores.unsqueeze(0).detach().cpu().numpy())
        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature
        if opt.use_ssim > 0.0:
            anomaly_score += opt.use_ssim * (1 - ssim(real_img, fake_img, data_range=1.0, size_average=True))

        z_distance = criterion(fake_z, real_z)

        with open("results/score.csv", "a") as f:
            f.write(f"{label.item()},{img_distance},"
                    f"{anomaly_score},{z_distance}\n")

    cal_pro_scores(masks, amaps)


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize([opt.img_size] * 2),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opt.img_size, opt.img_size))
    ])
    # mvtec_ad = MVTecAD(".", opt.dataset_name, train=False, transform=transform,
    #                  download=True)
    mvtec_ad = MVTecADDataset(root=".", target=opt.dataset_name, transforms=transform, mask_transforms=mask_transform,
                              train=False)
    test_dataloader = DataLoader(mvtec_ad, batch_size=1, shuffle=False)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    test_anomaly_detection(opt, generator, discriminator, encoder,
                           test_dataloader, device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str,
                        choices=MVTecAD.available_dataset_names,
                        help="name of MVTec Anomaly Detection Datasets")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--gaussian_blur", action="store_true",
                        help="perform blur both images before scoring")
    parser.add_argument("--gaussian_kernel_size", type=int, default=5,
                        help="blur kernel")
    parser.add_argument("--gaussian_blur_sigma1", type=float, default=0.1,
                        help="blur sigma1")
    parser.add_argument("--gaussian_blur_sigma2", type=float, default=5,
                        help="blur sigma2")
    parser.add_argument("--use_ssim", type=float, default=0.0,
                        help="add ssim loss in detection")
    opt = parser.parse_args()

    main(opt)

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision4ad.datasets import MVTecAD

from fanogan.train_encoder_izif import train_encoder_izif

from model import Generator, Discriminator, Encoder


def main(opt):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.Resize([opt.img_size]*2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])
    mvtec_ad = MVTecAD(".", opt.dataset_name, train=True, transform=transform,
                       download=True)
    if (opt.dataset_interval > 0.0):
        subset_size = int(opt.dataset_interval * mvtec_ad.__len__())
        mvtec_ad = Subset(mvtec_ad, range(subset_size))
    train_dataloader = DataLoader(mvtec_ad, batch_size=opt.batch_size,
                                  shuffle=True)

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    train_encoder_izif(opt, generator, discriminator, encoder,
                       train_dataloader, device)


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
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="number of training steps for "
                             "discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    parser.add_argument("--dataset_interval", type=float, default=1.0,
                        help="subset of training dataset")
    parser.add_argument("--encoder_denoise_level", type=float, default=0.0,
                        help="gaussian noise level")
    parser.add_argument("--encoder_inpainting", type=bool, default=False,
                        help="perform inpainting task")
    parser.add_argument("--aux_recon", action="store_true",
                        help="perform recon objective for the auxiliary task")

    opt = parser.parse_args()

    main(opt)

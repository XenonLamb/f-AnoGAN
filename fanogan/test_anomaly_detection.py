import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

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

    for (img, label) in tqdm(dataloader):

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)
        if(opt.gaussian_blur):
            blurrer = transforms.GaussianBlur((7,7),(opt.gaussian_blur_sigma1, opt.gaussian_blur_sigma2))
            fake_img = blurrer(fake_img)
            real_img = blurrer(real_img)
        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

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

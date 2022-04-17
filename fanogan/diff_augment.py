import torch
import torch.nn.functional as F

from torch.distributions import Bernoulli


def DiffAugment(x, policy='color,translation,cutout', channels_first=True):
    # print(x.shape)
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()

    return x


def rand_brightness(x, aug_prob=0.5):
    # print(x.shape)
    b = Bernoulli(aug_prob)
    aug_vec = b.sample((x.shape[0], 1, 1, 1)).to(x.device)
    keep_vec = 1 - aug_vec
    # print(x.shape, aug_vec.shape,keep_vec.shape)
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * aug_vec
    # print(x.shape, aug_vec.shape,keep_vec.shape)
    return x


def rand_saturation(x, aug_prob=0.5):
    b = Bernoulli(aug_prob)
    aug_vec = b.sample((x.shape[0], 1, 1, 1)).to(x.device)
    keep_vec = 1 - aug_vec

    x_mean = x.mean(dim=1, keepdim=True)
    x = x * keep_vec + (
                (x - x_mean) * (torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean) * aug_vec
    return x


def rand_contrast(x, aug_prob=0.5):
    b = Bernoulli(aug_prob)
    aug_vec = b.sample((x.shape[0], 1, 1, 1)).to(x.device)
    keep_vec = 1 - aug_vec

    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = x * keep_vec + ((x - x_mean) * (
                torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean) * aug_vec
    return x


def rand_translation(x, ratio=0.125, aug_prob=0.5):
    b = Bernoulli(aug_prob)
    aug_vec = b.sample((x.shape[0], 1, 1, 1)).to(x.device)
    keep_vec = 1 - aug_vec

    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.shape[0], 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.shape[0], 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x * keep_vec + (
        x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)) * aug_vec
    return x


def rand_cutout(x, ratio=0.5, aug_prob=0.5):
    b = Bernoulli(aug_prob)
    aug_vec = b.sample((x.shape[0], 1, 1, 1)).to(x.device)
    keep_vec = 1 - aug_vec
    # print(x.shape,  keep_vec.shape, aug_vec.shape)

    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    # print(x.shape, mask.shape, keep_vec.shape, aug_vec.shape)
    x = x * keep_vec + (x * mask.unsqueeze(1)) * aug_vec
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
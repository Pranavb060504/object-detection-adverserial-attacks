import torch
import numpy as np
import cv2

def get_patch_coords(img_shape, row, col, patch_size):
    _, H, W = img_shape
    y1 = row * patch_size
    y2 = min((row + 1) * patch_size, H)
    x1 = col * patch_size
    x2 = min((col + 1) * patch_size, W)
    return y1, y2, x1, x2

def apply_gaussian_noise(img, row, col, patch_size=16, std=0.05):
    y1, y2, x1, x2 = get_patch_coords(img.shape, row, col, patch_size)
    noise = torch.randn_like(img[:, y1:y2, x1:x2]) * std
    img[:, y1:y2, x1:x2] = torch.clamp(img[:, y1:y2, x1:x2] + noise, 0, 1)
    return img

def apply_salt_and_pepper_noise(img, row, col, patch_size=16, prob=0.05):
    y1, y2, x1, x2 = get_patch_coords(img.shape, row, col, patch_size)
    patch = img[:, y1:y2, x1:x2]
    rand = torch.rand_like(patch)
    patch[rand < (prob / 2)] = 0.0
    patch[rand > 1 - (prob / 2)] = 1.0
    img[:, y1:y2, x1:x2] = patch
    return img

def apply_gaussian_blur(img, row, col, patch_size=16):
    return apply_cv2_filter(img, row, col, patch_size, lambda x: cv2.GaussianBlur(x, (3, 3), 0))

def apply_median_blur(img, row, col, patch_size=16):
    return apply_cv2_filter(img, row, col, patch_size, lambda x: cv2.medianBlur(x, 3))

def apply_sharpen(img, row, col, patch_size=16):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return apply_cv2_filter(img, row, col, patch_size, lambda x: cv2.filter2D(x, -1, kernel))

def apply_brighten(img, row, col, patch_size=16, factor=1.1):
    y1, y2, x1, x2 = get_patch_coords(img.shape, row, col, patch_size)
    img[:, y1:y2, x1:x2] = torch.clamp(img[:, y1:y2, x1:x2] * factor, 0, 1)
    return img

def apply_darken(img, row, col, patch_size=16, factor=0.9):
    return apply_brighten(img, row, col, patch_size, factor)

def apply_contrast_increase(img, row, col, patch_size=16, factor=1.1):
    y1, y2, x1, x2 = get_patch_coords(img.shape, row, col, patch_size)
    patch = img[:, y1:y2, x1:x2]
    mean = patch.mean(dim=(1, 2), keepdim=True)
    img[:, y1:y2, x1:x2] = torch.clamp((patch - mean) * factor + mean, 0, 1)
    return img

def apply_contrast_decrease(img, row, col, patch_size=16, factor=0.9):
    return apply_contrast_increase(img, row, col, patch_size, factor)


def apply_cv2_filter(img, row, col, patch_size, filter_fn):
    y1, y2, x1, x2 = get_patch_coords(img.shape, row, col, patch_size)
    patch = img[:, y1:y2, x1:x2].clone()
    patch_np = (patch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    filtered = filter_fn(patch_np)
    filtered_tensor = torch.from_numpy(filtered.astype(np.float32) / 255).permute(2, 0, 1)
    img[:, y1:y2, x1:x2] = filtered_tensor.to(img.device)
    return img

attack_fn_map = {
    0: apply_gaussian_noise,
    1: apply_salt_and_pepper_noise,
    3: apply_gaussian_blur,
    4: apply_median_blur,
    5: apply_sharpen,
    6: apply_brighten,
    7: apply_darken,
    8: apply_contrast_increase,
    9: apply_contrast_decrease,
}

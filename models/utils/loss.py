import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def calculate_mae(img1: Tensor, img2: Tensor) -> Tensor:
    return torch.mean(torch.abs(img1 - img2), dim=[1, 2, 3])

def calculate_mse(img1: Tensor, img2: Tensor) -> Tensor:
    return torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    
def calculate_rmse(img1: Tensor, img2: Tensor):
    RMSE = torch.sqrt(calculate_mse(img1, img2))
    return RMSE.mean()

def calculate_mrae(img1: Tensor, img2: Tensor):
    MRAE = torch.mean(torch.abs(img1 - img2) / (img1 + 1e-9), dim=[1, 2, 3])
    return MRAE.mean()

def calculate_psnr(img1, img2, pixel_max = 1.):
    mse = calculate_mse(img1, img2)
    mse[mse==0] = 1e-9
    PSNR = 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return PSNR.mean()

def calculate_sam(x: Tensor, y: Tensor):
    assert x.shape == y.shape
    B = x.shape[0]
    C = x.shape[1]
    x = x.view(B, C, -1)
    y = y.view(B, C, -1)
    x_norm = torch.sum(torch.square(x), dim=1)
    y_norm = torch.sum(torch.square(y), dim=1)
    deno = torch.sqrt(x_norm * y_norm)
    mole = torch.sum(x * y, dim=1)
    sam = torch.acos(mole / (deno + 1e-10)).mean(dim=1)
    return sam.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel, mu=1.5):
    _1D_window = gaussian(window_size, mu).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIM(nn.Module):
    def __init__(self, channel, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", create_window(window_size, self.channel))

    def forward(self, img1, img2):
        size = img1.size()
        assert self.channel == size[-3]
        
        channel = self.channel
        window = self.window
        window_size = self.window_size

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()
    
class TVLoss(nn.Module):
    # Total Variation loss
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.mean(torch.diff(x, dim=-1)**2) + torch.mean(torch.diff(x, dim=-2)**2)
    
class Blur(nn.Module):
    def __init__(self, sigma=1, size=7, beta=0.25):
        super(Blur, self).__init__()
        self.size = size
        self.register_buffer("kernel", self.gen_LoG_kernel(sigma, size))# Laplacian of Gaussian
        self.beta = beta

    def forward(self, img: Tensor):
        B, C, H, W = img.shape
        img_lap = F.conv2d(img, self.kernel.expand(C, 1, self.size, self.size), padding='same', groups=C)
        blur_loss = - torch.log (torch.sum(img_lap ** 2, dim=[1, 2, 3]) / (H*W - torch.mean(img, dim=[1,2,3])**2) + 1e-8)
        return blur_loss.mean() * self.beta

    def gen_LoG_kernel(self, sigma, size):
        X = np.arange(size//2, -size//2, -1)
        Y = np.arange(size//2, -size//2, -1)
        xx, yy = np.meshgrid(X, Y)
        LoG_kernel = 1 / (np.pi * sigma ** 4) * (1 - (xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * np.exp(- (xx ** 2 + yy ** 2) / (2 * sigma ** 2))    
        return torch.from_numpy(LoG_kernel).type(torch.float32).view(1, 1, size, size)

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, img1, img2):
        f1 = torch.fft.rfftn(img1, dim=(-3,-2,-1))
        f2 = torch.fft.rfftn(img2, dim=(-3,-2,-1))
        return self.l1(f1, f2)

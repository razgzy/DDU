import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
import cv2
import math
import numpy as np


def block_inverse(HH:Tensor, gamma:Tensor):
    """
    :param HH: [1, H, W, 3, 3]
    :param gamma: [B, 1, 1, 1, 1]
    :return: X: [1, H, W, 3, 3]
    """
    
    A = HH / gamma # gamma^-1 Phi
    b, h, w, _, _ = A.shape
    A11_inv = 1 / (A[..., 0, 0] + 1) # b, h, w
    A12 = A[..., 0, 1] # b, h, w
    A21 = A[..., 1, 0] # b, h, w
    A22 = A[..., 1, 1] + 1 # b, h, w
    num1 = A11_inv * A12 # b, h, w
    num2 = A21 * A11_inv # b, h, w
    denom = 1 / (A22 - A21 * num1) # b, h, w
    B11_inv = torch.zeros(b, h, w, 2, 2, dtype=A.dtype, device=A.device)
    B11_inv[..., 0, 0] = A11_inv + num1 * denom * num2
    B11_inv[..., 0, 1] = - num1 * denom
    B11_inv[..., 1, 0] = - denom * num2     
    B11_inv[..., 1, 1] = denom 
    B12 = A[..., :2, 2:] # b, h, w, 2, 1
    B21 = A[..., 2:, :2] # b, h, w, 1, 2
    B22 = A[..., 2:, 2:] + 1 # b, h, w, 1, 1
    num1 = torch.matmul(B11_inv, B12) # b, h, w, 2, 1
    num2 = torch.matmul(B21, B11_inv) # b, h, w, 1, 2
    denom = 1 / (B22 - torch.matmul(B21, num1)) # b, h, w, 1, 1
    A_inv = torch.zeros_like(A)
    A_inv[..., :2, :2] = B11_inv + torch.matmul(num1, torch.matmul(denom, num2))
    A_inv[..., :2, 2:] = - torch.matmul(num1, denom)
    A_inv[..., 2:, :2] = - torch.matmul(denom, num2)
    A_inv[..., 2:, 2:] = denom
    return A_inv

def select_init_network(method, in_dim, out_dim, dim, n_blocks):
    method = method.lower()
    if method == 'mirnetv2':
        from .MIRNetV2 import MIRNet_v2
        return MIRNet_v2(in_dim=in_dim, out_dim=out_dim, dim=dim, n_RRG=1, n_MRB=n_blocks)
    elif method == 'mst':
        from .MST_Plus_Plus import MST_Plus_Plus
        return MST_Plus_Plus(in_dim=in_dim, out_dim=out_dim, dim=dim, stage=1)
    else:
        raise ValueError(f"method {method} not supported")
        
def select_denoise_network(method, in_dim, out_dim, dim, n_blocks):
    method = method.lower()
    if method == 'mirnetv2':
        from .MIRNetV2 import RRGModule as RRG
        return RRG(in_dim=in_dim, out_dim=out_dim, dim=dim, n_MRB=n_blocks)
    elif method == 'mst':
        from .MST_Plus_Plus import MSTModule as MST
        return MST(in_dim=in_dim, out_dim=out_dim, dim=dim)
    else:
        raise ValueError(f"method {method} not supported")
          
class DDU(nn.Module):

    def __init__(self, out_dim, deblur, reuse=False, cumsum=True, use_alpha=True, use_zeta=True, use_sigma=True, supervise_init=True, supervise_all=False):
        super().__init__()
        self.wvl_num = out_dim
        self.n_iters_deblur = deblur['n_iters']
        self.reuse = reuse
        self.use_alpha = use_alpha
        self.use_zeta = use_zeta
        self.use_sigma = use_sigma
        self.supervise_init = supervise_init
        self.supervise_all = supervise_all
        if self.supervise_all:
            self.supervise_init = True
        self.cumsum = cumsum
        if self.cumsum:
            gammas = torch.ones(self.n_iters_deblur-1)*0.005
            gammas[0] = 0.01
            self.gammas = nn.Parameter(gammas)
        else:
            self.gammas = nn.Parameter(torch.ones(self.n_iters_deblur-1)*0.01)
        if self.use_sigma:
            self.sigma_div_gammas = nn.Parameter(torch.ones(self.n_iters_deblur-1)*1.0)
        if self.use_zeta and self.use_alpha:
            self.zetas = nn.Parameter(torch.ones(self.n_iters_deblur-1)*1.0)

        self.init_network = select_init_network(deblur['name'], 3, self.wvl_num, deblur['dim'], deblur['n_blocks'])
        
        if self.use_sigma:
            in_dim = self.wvl_num + 1
        else:
            in_dim = self.wvl_num
        if self.reuse:
            deblur_network = select_denoise_network(deblur['name'], in_dim, self.wvl_num, deblur['dim'], deblur['n_blocks'])
            self.denoiser = deblur_network
        else:
            self.denoisers = nn.ModuleList([])
            for _ in range(self.n_iters_deblur - 1):
                self.denoisers.append(
                    select_denoise_network(deblur['name'], in_dim, self.wvl_num, deblur['dim'], deblur['n_blocks'])
                )
        self.plt_W_U = False
        self.save_img = False
        self.output_each_stage = False
        self.A_inv = None

    @property
    def hypa(self):
        hypa = {}
        if self.cumsum:
            gammas = torch.cumsum(torch.abs(self.gammas), dim=0)
        else:
            gammas = torch.abs(self.gammas)
        gammas = gammas.view(1, self.n_iters_deblur-1, 1, 1, 1, 1)
        hypa.update(gammas=gammas)
        if self.use_sigma:
            sigma_div_gammas = self.sigma_div_gammas.view(1, self.n_iters_deblur-1, 1, 1)
            hypa.update(sigma_div_gammas=sigma_div_gammas)
        if self.use_zeta and self.use_alpha:
            zetas = self.zetas.view(1, self.n_iters_deblur-1, 1, 1)
            hypa.update(zetas=zetas)
        return hypa

    def init_A_inv(self, HH):
        hypa = self.hypa
        gammas = hypa['gammas'].squeeze(0)
        self.A_inv = block_inverse(HH, gammas)
        
    def forward(self, y:Tensor, H:Tensor, H_herm:Tensor, HH:Tensor, **kwargs): 
        """
        :param y: [B, 3, H, W]
        :param H: [1, H, W/2+1, 3, N]
        :param H_herm: [1, H, W/2+1, N, 3]
        :param HH: [1, H, W/2+1, 3, 3]
        :return: X: [B, N, H, W]
        """
        b, _, h, w = y.shape
        # w1 = w//2+1
        V = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(y, dim=(-2, -1)), dim=(-2, -1)), dim=(-2)) # b, 3, h, w/2
        out = self.init_network(y)
        if self.supervise_init:
            out_dict = {}
            for key, value in out.items():
                out_dict[key] = [value]
        Z = out['out']

        # cv2.imwrite('Z1_500.png', (Z[:, 2, 570:738, 20:188].squeeze(0).cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('Z1_600.png', (Z[:, 12, 570:738, 20:188].squeeze(0).cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('Z1_700.png', (Z[:, 22, 570:738, 20:188].squeeze(0).cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('Z1_800.png', (Z[:, 32, 570:738, 20:188].squeeze(0).cpu().numpy()*255).astype(np.uint8))
        if self.plt_W_U:
            plt.figure('Z0')
            grid_summary = torchvision.utils.make_grid(Z[0, 2:-1:20, 300:556, 300:556].detach().unsqueeze(1), nrow=1, pad_value=1)
            plt.imshow(grid_summary.permute(1, 2, 0).cpu().float().numpy())
            # plt.imsave('Z0.png', grid_summary.permute(1, 2, 0).cpu().float().numpy())
            cv2.imwrite('Z0.png', (grid_summary.clip(0, 1).permute(1, 2, 0).float().cpu().numpy()*255).astype(np.uint8))

        alpha = 0

        hypa = self.hypa
        gammas = hypa['gammas']
        if self.use_sigma:
            sigma_div_gammas = hypa['sigma_div_gammas'].to(Z.dtype)
        if self.use_zeta and self.use_alpha:
            zetas = hypa['zetas'].to(Z.dtype)

        # deblur
        V = V.permute(0, 2, 3, 1).unsqueeze(-1) # b, h, w1, 3, 1
        if self.output_each_stage:
            out_list = []
        for i in range(self.n_iters_deblur - 1):
            if self.use_alpha:
                tilde_I = Z - alpha # b, n, h, w
            else:
                tilde_I = Z # b, n, h, w
            tilde_U = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(tilde_I.float(), dim=(-2, -1)), dim=(-2, -1)), dim=(-2)) # b, n, h, w1
            tilde_U = tilde_U.permute(0, 2, 3, 1).unsqueeze(-1) # b, h, w1, n, 1
            gamma = gammas[:, i] # b, 1, 1, 1, 1
            if self.A_inv is None:
                A_inv = block_inverse(HH, gamma)  # b, h, w1, 3, 3
            else:
                A_inv = self.A_inv[i:i+1]
            # A_inv = torch.rand(1, h, w//2 + 1, 3, 3, dtype=torch.complex64, device=H.device)
            U = tilde_U + torch.matmul(H_herm, V - torch.matmul(A_inv, torch.matmul(HH, V) / gamma + torch.matmul(H, tilde_U))) / gamma # b, h, w/2, n, 1
            U = U.squeeze(-1).permute(0, 3, 1, 2) # b, n, h, w1
            I = torch.fft.fftshift(torch.fft.irfft2(torch.fft.ifftshift(U, dim=(-2)), dim=(-2, -1)), dim=(-1, -2)).to(tilde_I.dtype) # b, n, h, w
            if self.plt_W_U:
                grid_summary = torchvision.utils.make_grid(I[0, 2:-1:20, 300:556, 300:556].detach().unsqueeze(1), nrow=1, pad_value=1)
                plt.figure('I{}'.format(i+1))
                plt.imshow(grid_summary.permute(1, 2, 0).cpu().float().numpy())
                # plt.imsave('I{}.png'.format(i+1), grid_summary.permute(1, 2, 0).cpu().float().numpy())
                cv2.imwrite('I{}.png'.format(i+1), (grid_summary.clip(0, 1).permute(1, 2, 0).float().cpu().numpy()*255).astype(np.uint8))

            if self.use_alpha:
                tilde_Z = I + alpha # b, n, h, w
            else:
                tilde_Z = I # b, n, h, w
                
            if i == self.n_iters_deblur - 1:
                mo = True
            else:
                mo = False

            if self.use_sigma:
                input = torch.cat([tilde_Z , sigma_div_gammas[:, i:i+1].expand(b, 1, h, w)], dim = 1) # b, n+1, h, w
            else:
                input = tilde_Z
                
            if self.reuse:
                out = self.denoiser(input, mo=mo) # b, n, h, w
            else:
                out = self.denoisers[i](input, mo=mo) # b, n, h, w
            
            if self.supervise_all:
                for key, value in out.items():
                    out_dict[key].append(value)

            Z = out['out']
            if self.output_each_stage:
                out_list.append(Z)
            if i < self.n_iters_deblur - 1:
                if self.use_alpha:
                    if self.use_zeta:
                        alpha = alpha + zetas[:, i:i+1] * (I - Z) # b, n, h, w
                    else:
                        alpha = alpha + I - Z # b, n, h, w

            if self.plt_W_U:
                plt.figure('Z{}'.format(i+1))
                grid_summary = torchvision.utils.make_grid(Z[0, 2:-1:20, 300:556, 300:556].detach().unsqueeze(1), nrow=1, pad_value=1)
                plt.imshow(grid_summary.permute(1, 2, 0).cpu().float().numpy())
                # plt.imsave('Z{}.png'.format(i+1), grid_summary.permute(1, 2, 0).float().cpu().numpy())
                cv2.imwrite('Z{}.png'.format(i+1), (grid_summary.clip(0, 1).permute(1, 2, 0).float().cpu().numpy()*255).astype(np.uint8))
        
        if self.plt_W_U:
            print(gammas[0])
            print(sigma_div_gammas[0])
            print(zetas[0])
            plt.show()
        
        if self.supervise_all:
            for key, value in out.items():
                out_dict[key] = torch.stack(out_dict[key], dim=1)
        elif self.supervise_init:
            for key, value in out.items():
                out_dict[key].append(value)
                out_dict[key] = torch.stack(out_dict[key], dim=1)

        if self.output_each_stage:
            return out_list
        elif self.supervise_init or self.supervise_all:
            return out_dict
        else:
            return out
    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = set()
        for name, module in self.named_modules():
            if name != '' and hasattr(module, 'no_weight_decay'):
                no_decay.update({f"{name}.{param}" for param in module.no_weight_decay()})
        return no_decay
    
    @torch.jit.ignore
    def save_W0(self, W):
        max = W.abs().max()
        F500W0_color = cv2.applyColorMap(((W[0,2].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        F600W0_color = cv2.applyColorMap(((W[0,12].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        F700W0_color = cv2.applyColorMap(((W[0,22].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        F800W0_color = cv2.applyColorMap(((W[0,32].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        cv2.imwrite('500W0.png', F500W0_color[..., ::-1])
        cv2.imwrite('600W0.png', F600W0_color[..., ::-1])
        cv2.imwrite('700W0.png', F700W0_color[..., ::-1])
        cv2.imwrite('800W0.png', F800W0_color[..., ::-1])

    @torch.jit.ignore
    def save_U1(self, U):
        max = U.abs().max()
        Fb500U1_color = cv2.applyColorMap(((U[0,2].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        Fb600U1_color = cv2.applyColorMap(((U[0,12].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        Fb700U1_color = cv2.applyColorMap(((U[0,22].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        Fb800U1_color = cv2.applyColorMap(((U[0,32].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        cv2.imwrite('500U1.png', Fb500U1_color[..., ::-1])
        cv2.imwrite('600U1.png', Fb600U1_color[..., ::-1])
        cv2.imwrite('700U1.png', Fb700U1_color[..., ::-1])
        cv2.imwrite('800U1.png', Fb800U1_color[..., ::-1])

    @torch.jit.ignore
    def save_W1(self, W):
        max = W.abs().max()
        F500W1_color = cv2.applyColorMap(((W[0,2].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        F600W1_color = cv2.applyColorMap(((W[0,12].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        F700W1_color = cv2.applyColorMap(((W[0,22].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        F800W1_color = cv2.applyColorMap(((W[0,32].abs() / max).pow(1/3).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        cv2.imwrite('500W1.png', F500W1_color[..., ::-1])
        cv2.imwrite('600W1.png', F600W1_color[..., ::-1])
        cv2.imwrite('700W1.png', F700W1_color[..., ::-1])
        cv2.imwrite('800W1.png', F800W1_color[..., ::-1])
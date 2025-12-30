from typing import List, Dict, Tuple
import lightning as L
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from models.utils.loss import SSIM
from models.utils.optims_group import set_weight_decay
from models.utils.EMA import AveragedModel, get_ema_multi_avg_fn
import torchvision
from models.utils.loss import calculate_psnr, calculate_sam, calculate_rmse, calculate_mrae
import time
import cv2
import hdf5storage
import os
import numpy as np
import logging
import hdf5storage
from matplotlib import pyplot as plt
from models.networks.SFDUMTV2 import block_inverse

class NetworkModel(L.LightningModule):
    def __init__(self, network: torch.nn.Module, optimizer_dict: Dict, schedular_dict: Dict, accumulate_grad_batches = 1, ssim_loss=False, ema=False, crop_size=(128, 128), val_crop_size=(512, 512), test_crop_size=(512, 512), blind=False, half_blind=False, gaussian_noise=0.0025, shot_noise_bit=14, cyclic_conv=False, no_crop=False):
        super().__init__()
        self.save_hyperparameters()
        self.network = network
        self.criterion = nn.ModuleDict({'l1': nn.L1Loss(), 'ssim': SSIM(43)})
        self.optimizer_dict = optimizer_dict
        self.schedular_dict = schedular_dict
        self.automatic_optimization = True
        self.accumulate_grad_batches = accumulate_grad_batches
        self.step = 0
        self.ssim_loss = ssim_loss
        self.ema = ema
        if self.ema:
            self.ema_decay = 0.999
            self.ema_network = AveragedModel(self.network, multi_avg_fn=get_ema_multi_avg_fn())
            for param in self.ema_network.parameters():
                param.requires_grad = False
            # self.ema_network.eval()
        self.gaussian_noise = gaussian_noise
        self.shot_noise_bit = shot_noise_bit
        self.blind = blind
        self.half_blind = half_blind
        self.cyclic_conv = cyclic_conv
        self.no_crop = no_crop
        cPSF = torch.tensor(hdf5storage.loadmat('optics/cpsfs.mat')['cpsf']) # 3, N, H, W
        print(cPSF.shape)
        self.register_buffer('PSF', cPSF.unsqueeze(0))
        self.register_buffer('PSF_flip', cPSF[..., 1:, 1:].flip(-1).flip(-2))
        self.test_crop_size = test_crop_size
        if not self.blind or self.half_blind:
            pad_h = (val_crop_size[0] - cPSF.shape[-2]) // 2
            pad_w = (val_crop_size[1] - cPSF.shape[-1]) // 2
            val_PSF = F.pad(cPSF, (pad_w, pad_w, pad_h, pad_h))
            self.val_H = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(val_PSF, dim=(-2, -1)), dim=(-2, -1)), dim=(-2)).permute(2,3,0,1).unsqueeze(0) # 1, H, W/2+1, 3, N
            self.val_H_herm = self.val_H.conj().transpose(-2, -1)
            self.val_HH = torch.matmul(self.val_H, self.val_H_herm)

            pad_h = (crop_size[0] - cPSF.shape[-2]) // 2
            pad_w = (crop_size[1] - cPSF.shape[-1]) // 2
            train_PSF = F.pad(cPSF, (pad_w, pad_w, pad_h, pad_h))
            train_H = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(train_PSF, dim=(-2, -1)), dim=(-2, -1)), dim=(-2)).permute(2,3,0,1).unsqueeze(0) # 1, H, W/2+1, 3, N
            train_H_herm = train_H.conj().transpose(-2, -1)
            train_HH = torch.matmul(train_H, train_H_herm)
            self.register_buffer('train_H', train_H)
            self.register_buffer('train_H_herm', train_H_herm)
            self.register_buffer('train_HH', train_HH)
        
        self.strict_loading = True

    def configure_optimizers(self):
        params = []
        net_param, no_decay_names= set_weight_decay(self.network)
        params.extend(net_param)
        logger = logging.getLogger("lightning.pytorch.core")
        logger.info(f"No weight decay list: {no_decay_names}")
        if 'name' in self.optimizer_dict.keys():
            optimizer_name = self.optimizer_dict['name']
        else:
            optimizer_name = 'AdamW'
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr = self.optimizer_dict['lr'])
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr = self.optimizer_dict['lr'], weight_decay=self.optimizer_dict['weight_decay'])
        
        interval = self.schedular_dict['interval']
        if interval == 'epoch':
            T_0 = self.trainer.max_epochs
        else:
            # T_0 = self.trainer.max_steps
            iter_per_epoch = self.trainer.datamodule.step_per_epoch
            T_0 = self.trainer.max_epochs * iter_per_epoch
        print('T_0:', T_0)
        # lr_scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=self.schedular_dict['eta_min'])
        lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=self.schedular_dict['eta_min']*self.optimizer_dict['lr'])
        lr_scheduler1_config = {
            "scheduler": lr_scheduler1,
            "interval": interval,
            "name": optimizer_name
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler1_config}
    
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        if self.automatic_optimization:
            self.step = self.global_step
        if self.ema:
            decay = self.ema_decay * (1 - math.exp(-self.step / 2000))
            self.log('ema_decay', decay)
            self.ema_network.update_parameters(self.network, decay)

    def optimize(self, batch_idx, total_loss):
        if (batch_idx) % self.accumulate_grad_batches == 0:
            opts = self.optimizers()
            if isinstance(opts, List):
                for opt in opts:
                    opt.zero_grad()
            else:
                opts.zero_grad()

        self.manual_backward(total_loss / self.accumulate_grad_batches)

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.step += 1
            opts = self.optimizers()
            if isinstance(opts, List):
                for opt in opts:
                    opt.step()
            else:
                opts.step()
            if self.ema:
                decay = self.ema_decay * (1 - math.exp(-self.step / 2000))
                self.log('ema_decay', decay)
                self.ema_network.update_parameters(self.network, decay)

        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            schs = self.lr_schedulers()
            if isinstance(schs, List):
                for sch in schs:
                    sch.step()
            else:
                schs.step()
    
    def forward(self, input) -> Tensor:
        outputs = self.network(**input)

        return outputs
    
    def forward_ema(self, input) -> Tensor:
        outputs = self.ema_network(**input)

        return outputs

    def calculate_HH(self, H: torch.Tensor):
        """
        :param H: [1, h, w/2+1, 3, N]
        """
        H_ht = H.conj().permute(0,1,2,4,3)
        HH = torch.matmul(H, H_ht)
        return HH

    @torch.no_grad()
    def image_formation(self, spectrum):
        _, C, h, w = spectrum.shape
        psfs = self.PSF
        _, _, C_psf, H_psf, W_psf = psfs.shape
        assert C == C_psf
        pad_h = (h - H_psf) // 2
        pad_w = (w - W_psf) // 2
        if pad_h or pad_w: 
            psfs = F.pad(psfs, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0) 
        F_psfs = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(psfs, dim=(-2, -1)), dim=(-2, -1)), dim=(-2))
        F_spectrum = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)), dim=(-2)).unsqueeze(1)
        rgb_encoded = torch.fft.fftshift(torch.fft.irfft2(torch.fft.ifftshift(F_psfs * F_spectrum, dim=(-2)), dim=(-2, -1)), dim=(-1, -2)).sum(dim=2)
        if self.cyclic_conv:
            return rgb_encoded
        else:
            return rgb_encoded[..., 20:-20, 20:-20]
        
    @torch.no_grad()
    def image_formation_conv(self, spectrum):
        _, C, h, w = spectrum.shape
        _, C_psf, H_psf, W_psf = self.PSF_flip.shape
        assert C == C_psf
        spectrum = torch.nn.functional.pad(spectrum, (W_psf // 2, W_psf // 2, H_psf // 2, H_psf // 2), mode='circular')
        rgb_encoded = torch.nn.functional.conv2d(spectrum, self.PSF_flip, padding = 0)
        if self.cyclic_conv:
            return rgb_encoded
        else:
            return rgb_encoded[..., 20:-20, 20:-20]
    
    def add_gaussian_noise(self, coded):
        b, c, h, w = coded.shape
        assert c == 3
        noise = self.gaussian_noise * torch.randn((b, 4, h, w)).to(coded)
        noise = torch.stack([noise[:,0], (noise[:,1]+noise[:,2]) / 2, noise[:,3]], dim=1)
        return (coded + noise).clip(0,1)
    
    def add_poisson_noise(self, image, bit=14):
        image = image.clip(0, None)
        image = image * (2**bit - 1)
        noisy_image = torch.poisson(image)
        noisy_image = noisy_image / (2**bit - 1)
        return noisy_image.clip(0, 1)
    
    def quantization(self, image, bit=12):
        image = image * (2**bit - 1)
        image = torch.round(image)
        image = image / (2**bit - 1)
        return image

    def crop_gt(self, gt):
        if self.cyclic_conv:
            if not self.no_crop:
                gt = gt[..., 20:-20, 20:-20].contiguous()
        else:
            if not self.no_crop:
                gt = gt[..., 40:-40, 40:-40].contiguous()
            else:
                gt = gt[..., 20:-20, 20:-20].contiguous()
        return gt
    
    @torch.no_grad()
    def generate_coded(self, spectrum):
        with torch.autocast(device_type='cuda', enabled=False):
            coded = self.image_formation_conv(spectrum)
            if self.shot_noise_bit > 0:
                coded = self.add_poisson_noise(coded, self.shot_noise_bit)
            coded = self.add_gaussian_noise(coded)
            coded = self.quantization(coded)
        return coded
    
    def prepare_multi_resolution(self, outputs, gt_dict, num_scales):
        if 'out1' in outputs:
            spectrum_2x = F.interpolate(gt_dict['out'], scale_factor=0.5)
            gt_dict.update({'out1': spectrum_2x})
            if not self.no_crop:
                outputs['out1'] = outputs['out1'][..., 10:-10, 10:-10].contiguous()
            outputs['out1'] = outputs['out1'].clip(0, None)
            num_scales += 1
            if 'out2' in outputs:
                spectrum_4x = F.interpolate(spectrum_2x, scale_factor=0.5)
                gt_dict.update({'out2': spectrum_4x})
                if not self.no_crop:
                    outputs['out2'] = outputs['out2'][..., 5:-5, 5:-5].contiguous()
                outputs['out2'] = outputs['out2'].clip(0, None)
                num_scales += 1
        return outputs, gt_dict, num_scales
    
    def training_step(self, batch, batch_idx):
        spectrum = batch['spectrum']
        b, m, n, h, w = spectrum.shape
        spectrum = spectrum.view(b*m, n, h, w)
        coded = self.generate_coded(spectrum)
        spectrum = self.crop_gt(spectrum)
        losses = {}
        total_loss = torch.tensor(0).to(spectrum)
        name = 'train/'
        if not self.blind: 
            input = {'y': coded, 'H': self.train_H, 'H_herm': self.train_H_herm, 'HH': self.train_HH}    
        else:
            input = {'x': coded}
        outputs = self.forward(input)
        if not self.no_crop:
            outputs['out'] = outputs['out'][..., 20:-20, 20:-20].contiguous()
        outputs['out'] = outputs['out'].clip(0, None)
        gt_dict = {'out': spectrum}
        num_scales = 1
        outputs, gt_dict, num_scales = self.prepare_multi_resolution(outputs, gt_dict, num_scales)
        l1_spectrum_loss = 0
        ssim_spectrum_loss = 0
        with torch.autocast(device_type='cuda', enabled=False):
            for key, value in gt_dict.items():
                if len(outputs[key].shape) == 5:
                    l1_spectrum_loss = self.criterion['l1'](value.unsqueeze(1).expand_as(outputs[key]).float(), outputs[key].float())
                else:
                    l1_spectrum_loss += self.criterion['l1'](value.float(), outputs[key].float())
                if self.ssim_loss:
                    if len(outputs[key].shape) == 5:
                        b, m, n, h, w = outputs[key].shape
                        ssim_spectrum_loss += 1 - self.criterion['ssim'](value.unsqueeze(1).expand_as(outputs[key]).reshape(b*m, n, h, w).float(), outputs[key].reshape(b*m, n, h, w).float())
                    else:
                        ssim_spectrum_loss += 1 - self.criterion['ssim'](value.float(), outputs[key].float())
        l1_spectrum_loss /= num_scales
        ssim_spectrum_loss /= num_scales
        losses[name + 'l1_spectrum_loss'] = l1_spectrum_loss.item()
        if self.ssim_loss:
            losses[name + 'ssim_spectrum_loss'] = ssim_spectrum_loss.item()
            total_loss = 0.85 * ssim_spectrum_loss + 0.15 *l1_spectrum_loss
        else:
            total_loss = l1_spectrum_loss

        losses[name + 'loss'] = total_loss.item()
        self.log_dict(losses, on_epoch=False, on_step=True)
        self.log('loss', total_loss.item(), on_epoch=False, on_step=True, logger=False, prog_bar=True)
        
        if batch_idx == 0:
            # print(batch['id'])
            idx = 0
            self.logSpectralImages(spectrum[idx].cpu().detach(), outputs['out'][idx].cpu().detach())
            self.logCodedImages(coded[idx].cpu().detach())
            if not self.blind:
                self.loghypa()
        
        if not self.automatic_optimization:
            self.optimize(batch_idx, total_loss)

        return total_loss

    def on_validation_start(self):
        if not self.blind or self.half_blind:
            device = self.train_H.device
            self.val_H = self.val_H.to(device)
            self.val_H_herm = self.val_H_herm.to(device)
            self.val_HH = self.val_HH.to(device)
            if hasattr(self.network, 'init_A_inv'):
                self.network.init_A_inv(self.val_HH)
                if self.ema:
                    self.ema_network.module.init_A_inv(self.val_HH)

    def validation_metrices(self, gt, output):
        with torch.autocast(device_type='cuda', enabled=False):
            gt = gt.float()
            output = output.float()
            SAM = calculate_sam(gt, output)
            PSNR = calculate_psnr(gt, output)
            SSIM = self.criterion['ssim'](gt, output)
        return SAM, PSNR, SSIM

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # coded = batch['coded']
        # b, m, _, h, w = coded.shape
        # coded = coded.view(b*m, 3, h, w)
        spectrum = batch['spectrum']
        b, m, n, h, w = spectrum.shape
        spectrum = spectrum.view(b*m, n, h, w)
        b, n, h, w = spectrum.shape
        coded = self.generate_coded(spectrum)
        spectrum = self.crop_gt(spectrum)
        # plt.figure('GT')
        # grid_summary = torchvision.utils.make_grid(spectrum[0].detach().unsqueeze(1), nrow=8, pad_value=1)
        # plt.imshow(grid_summary.permute(1, 2, 0).cpu().numpy())
        losses = {}
        stage_name = 'valid/'
        name = stage_name + self.trainer.datamodule.val_dataset_name[dataloader_idx] + '/'
        if not self.blind:
            input = {'y': coded, 'H': self.val_H, 'H_herm': self.val_H_herm, 'HH': self.val_HH}
        else:
            input = {'x': coded}
        outputs = self.forward(input)
        if len(outputs['out'].shape) == 5:
            outputs['out'] = outputs['out'][:, -1]
        if not self.no_crop:
            outputs['out'] = outputs['out'][:, :, 20:-20, 20:-20].contiguous()
        outputs['out'] = outputs['out'].clip(0, None)
        SAM, PSNR, SSIM = self.validation_metrices(spectrum, outputs['out'])
        losses[name + 'sam'] = SAM.item()
        losses[name + 'psnr'] = PSNR.item()
        losses[name + 'ssim'] = SSIM.item()
        if self.ema:
            outputs = self.forward_ema(input)
            if len(outputs['out'].shape) == 5:
                outputs['out'] = outputs['out'][:, -1]
            if self.half_blind:
                output = outputs['out']
                gamma = 5e-1
                V = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(coded.float(), dim=(-2, -1)), dim=(-2, -1)), dim=(-2))
                V = V.permute(0, 2, 3, 1).unsqueeze(-1)
                W = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(output.float(), dim=(-2, -1)), dim=(-2, -1)), dim=(-2))
                tilde_U = W.permute(0, 2, 3, 1).unsqueeze(-1)
                A_inv = block_inverse(self.val_HH, gamma)
                U = tilde_U + torch.matmul(self.val_H_herm, V - torch.matmul(A_inv, torch.matmul(self.val_HH, V) / gamma + torch.matmul(self.val_H, tilde_U))) / gamma
                U = U.squeeze(-1).permute(0, 3, 1, 2)
                result = torch.fft.fftshift(torch.fft.irfft2(torch.fft.ifftshift(U, dim=(-2)), dim=(-2, -1)), dim=(-1, -2))
                outputs['out'] = result
            if not self.no_crop:
                outputs['out'] = outputs['out'][:, :, 20:-20, 20:-20].contiguous()
            outputs['out'] = outputs['out'].clip(0, None)
            SAM, PSNR, SSIM = self.validation_metrices(spectrum, outputs['out'])
            losses[name + 'sam_ema'] = SAM.item()
            losses[name + 'psnr_ema'] = PSNR.item()
            losses[name + 'ssim_ema'] = SSIM.item()
        self.log_dict(losses, add_dataloader_idx=False)

    def on_validation_end(self):
        metrics = self.trainer.callback_metrics
        dataset_dict = {}
        for key, value in metrics.items():
            if 'valid' in key:
                _, dataset_name, metric_name = key.split('/')
                dataset_name = dataset_name.split('_')[0]
                if dataset_name not in dataset_dict:
                    dataset_dict[dataset_name] = {'psnr': [], 'sam': [], 'ssim': []}
                if metric_name == 'psnr_ema':
                    dataset_dict[dataset_name]['psnr'].append(value)
                elif metric_name == 'sam_ema':
                    dataset_dict[dataset_name]['sam'].append(value)
                elif metric_name == 'ssim_ema':
                    dataset_dict[dataset_name]['ssim'].append(value)

        avg_psnr_list = []
        for key, value in dataset_dict.items():
            avg_psnr = sum(value['psnr']) / len(value['psnr'])
            avg_sam = sum(value['sam']) / len(value['sam'])
            avg_ssim = sum(value['ssim']) / len(value['ssim'])
            self.logger.experiment.add_scalar('valid_avg/' + key + '/psnr', avg_psnr, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid_avg/' + key + '/sam', avg_sam, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('valid_avg/' + key + '/ssim', avg_ssim, global_step=self.current_epoch)
            avg_psnr_list.append(avg_psnr)

        if len(avg_psnr_list):
            hp_metric = sum(avg_psnr_list) / len(avg_psnr_list)
            self.logger.experiment.add_scalar('hp_metric', hp_metric, global_step=self.current_epoch)
            self.trainer._logger_connector._callback_metrics.update({'hp_metric': hp_metric})
        
        if not self.blind:
            self.val_H = self.val_H.to('cpu')
            self.val_H_herm = self.val_H_herm.to('cpu')
            self.val_HH = self.val_HH.to('cpu')
            self.network.A_inv = None
            if self.ema:
                self.ema_network.module.A_inv = None

    def prepare_noblind_input(self, coded):
        _, _, H, W = coded.shape
        pad_h = (H - self.PSF.shape[-2]) // 2
        pad_w = (W - self.PSF.shape[-1]) // 2
        PSF = F.pad(self.PSF, (pad_w, pad_w, pad_h, pad_h))
        H = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(PSF, dim=(-2, -1)), dim=(-2, -1)), dim=(-2)).permute(0,3,4,1,2) # 1, H, W/2+1, 3, N
        H_herm = H.conj().transpose(-2, -1)
        HH = torch.matmul(H, H_herm)
        input = {'y': coded, 'H': H, 'H_herm': H_herm, 'HH': HH}
        return input
    


    def on_test_start(self):
        self.save_coded = False
        self.save_gt = False
        self.save_mat = False
        self.save_png = False
        self.wvls = range(480, 910, 10)
        self.folder = 'results'
        self.exp = '64Res5stg'
        if not self.blind or self.half_blind:
            cPSF = self.PSF.squeeze(0)
            pad_h = (self.test_crop_size[0] - cPSF.shape[-2]) // 2
            pad_w = (self.test_crop_size[1] - cPSF.shape[-1]) // 2
            test_PSF = F.pad(cPSF, (pad_w, pad_w, pad_h, pad_h))
            self.test_H = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(test_PSF, dim=(-2, -1)), dim=(-2, -1)), dim=(-2)).permute(2,3,0,1).unsqueeze(0) # 1, H, W/2+1, 3, N
            self.test_H_herm = self.test_H.conj().transpose(-2, -1)
            self.test_HH = torch.matmul(self.test_H, self.test_H_herm)
            if hasattr(self.network, 'init_A_inv'):
                if self.ema:
                    self.ema_network.module.init_A_inv(self.test_HH)
                else:
                    self.network.init_A_inv(self.test_HH)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        name = self.trainer.datamodule.test_dataset_name[dataloader_idx] + '/'
        folder = self.folder
        exp = self.exp
        if self.save_mat or self.save_png:
            path = folder+ '/' + exp + '/' + name
            if not os.path.exists(path):
                os.makedirs(path)
        id = batch['id'][0]
        spectrum = batch['spectrum']
        b, m, n, h, w = spectrum.shape
        spectrum = spectrum.view(b*m, n, h, w)
        coded = self.generate_coded(spectrum)
        spectrum = self.crop_gt(spectrum)
        b, c, h, w = coded.shape
        if not self.blind:
            input = self.prepare_noblind_input(coded)
        else:
            input = {'x': coded}
        start_time = time.time()
        if self.ema:
            outputs = self.forward_ema(input)
        else:
            outputs = self.forward(input)
        # print(torch.cuda.memory_allocated()/1024/1024)
        # print(torch.cuda.max_memory_allocated()/1024/1024)
        end_time = time.time()
        stage_name = 'test/'
        self.log(stage_name + name + 'time', end_time - start_time, add_dataloader_idx=False)
        if len(outputs['out'].shape) == 5:
            outputs['out'] = outputs['out'][:, -1]
        if self.half_blind:
            output = outputs['out']
            gamma = 5e-1
            V = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(coded.float(), dim=(-2, -1)), dim=(-2, -1)), dim=(-2))
            V = V.permute(0, 2, 3, 1).unsqueeze(-1)
            W = torch.fft.fftshift(torch.fft.rfft2(torch.fft.ifftshift(output.float(), dim=(-2, -1)), dim=(-2, -1)), dim=(-2))
            tilde_U = W.permute(0, 2, 3, 1).unsqueeze(-1)
            A_inv = block_inverse(self.test_HH, gamma)
            U = tilde_U + torch.matmul(self.test_H_herm, V - torch.matmul(A_inv, torch.matmul(self.test_HH, V) / gamma + torch.matmul(self.test_H, tilde_U))) / gamma
            U = U.squeeze(-1).permute(0, 3, 1, 2)
            result = torch.fft.fftshift(torch.fft.irfft2(torch.fft.ifftshift(U, dim=(-2)), dim=(-2, -1)), dim=(-1, -2))
            outputs['out'] = result
        if not self.no_crop:
            outputs['out'] = outputs['out'][:, :, 20:-20, 20:-20].contiguous()
        outputs['out'] = outputs['out'].clip(0, None)
        SAM, PSNR, SSIM = self.validation_metrices(spectrum, outputs['out'])
        self.log(stage_name + name + 'psnr', PSNR.item(), add_dataloader_idx=False)
        self.log(stage_name + name + 'ssim', SSIM.item(), add_dataloader_idx=False)
        # self.log(stage_name + name + 'rmse', RMSE.item(), add_dataloader_idx=False)
        self.log(stage_name + name + 'sam', SAM.item(), add_dataloader_idx=False)
        out = outputs['out'].squeeze(0).cpu().numpy()
        # self.write_excel([[id, PSNR.item(), SAM.item(), SSIM.item()]], self.exp + '-result.xlsx')

        if self.save_gt:
            gt = spectrum.float().squeeze(0).cpu().numpy()
            path = folder+ '/' + 'gt' + '/' + name + id + '/'
            hdf5storage.savemat(path + 'spectrum.mat', {'HSI': gt})
            for i in range(spectrum.shape[1]):
                save_path = path + '{:.0f}nm.png'.format(self.wvls[i])
                cv2.imwrite(save_path, (gt[i, :, :]*255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if self.save_coded:
            coded = cv2.cvtColor((coded[:, :, 20:-20, 20:-20].clip(0, 1).squeeze().permute(1,2,0).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            path = folder+ '/' + 'coded' + '/' + name
            cv2.imwrite(path + id + '.png', coded, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        max_spectrum = spectrum.max()
        if self.save_mat:
            out = outputs['out'].squeeze(0).cpu().numpy()
            hdf5storage.savemat(path + 'spectrum.mat', {'HSI': out})
        if self.save_png:
            if max_spectrum > 1.:
                out = (outputs['out'] / max_spectrum).squeeze(0).cpu().numpy()
            else:
                out = outputs['out'].squeeze(0).cpu().numpy()
            for i in range(spectrum.shape[1]):
                save_path = path + id+'_{:.0f}nm.png'.format(self.wvls[i]*1e9)
                cv2.imwrite(save_path, (out[i, :, :]*255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    def on_test_end(self):
        metrics = self.trainer.callback_metrics
        dataset_dict = {}
        for key, value in metrics.items():
            if 'test' in key:
                _, dataset_name, metric_name = key.split('/')
                dataset_name = dataset_name.split('_')[0]
                if dataset_name not in dataset_dict:
                    dataset_dict[dataset_name] = {'psnr': [], 'sam': [], 'ssim': []}
                if metric_name == 'psnr':
                    dataset_dict[dataset_name]['psnr'].append(value)
                elif metric_name == 'sam':
                    dataset_dict[dataset_name]['sam'].append(value)
                elif metric_name == 'ssim':
                    dataset_dict[dataset_name]['ssim'].append(value)
        Avg_PSNR = []
        Avg_SAM = []
        Avg_SSIM = []
        for key, value in dataset_dict.items():
            avg_psnr = sum(value['psnr']) / len(value['psnr'])
            avg_sam = sum(value['sam']) / len(value['sam'])
            avg_ssim = sum(value['ssim']) / len(value['ssim'])
            self.logger.experiment.add_scalar('test_avg/' + key + '/psnr', avg_psnr, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('test_avg/' + key + '/sam', avg_sam, global_step=self.current_epoch)
            self.logger.experiment.add_scalar('test_avg/' + key + '/ssim', avg_ssim, global_step=self.current_epoch)
            Avg_PSNR.append(avg_psnr)
            Avg_SAM.append(avg_sam)
            Avg_SSIM.append(avg_ssim)
            print(key, 'PSNR:', avg_psnr, 'SAM:', avg_sam, 'SSIM:', avg_ssim)
        
        print('Avg', 'PSNR:', ((Avg_PSNR[0] + 2 * Avg_PSNR[1]) / 3).item(), 'SAM:', ((Avg_SAM[0] + 2 * Avg_SAM[1]) / 3).item(), 'SSIM:', ((Avg_SSIM[0] + 2 * Avg_SSIM[1]) / 3).item())

    @torch.no_grad()
    def logSpectralImages(self, gt, outputs, name = 'visualize/'):
        if len(outputs.shape) == 4:
            summary = torch.cat([gt, outputs[-1]], dim=-1)
        else:
            summary = torch.cat([gt, outputs], dim=-1)
        if summary.max() > 1:
            summary = summary / summary.max()
        grid_summary = torchvision.utils.make_grid(summary[::2].unsqueeze(1), nrow=4, pad_value=1)
        self.logger.experiment.add_image(name + "spetrum", grid_summary.numpy(), self.step)
    
    @torch.no_grad()
    def logCodedImages(self, coded, name = 'visualize/'):
        coded = coded.expand(3, -1, -1)
        self.logger.experiment.add_image(name + "coded", coded.numpy(), self.step)
    
    @torch.no_grad()
    def loghypa(self):
        hypa = self.network.hypa
        for key, value in hypa.items():
            fig = plt.figure(figsize=(4,3))
            plt.plot(value[0].squeeze().detach().cpu().numpy())
            fig.canvas.draw()
            matrix = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            self.logger.experiment.add_image('hypa/'+key, matrix, self.step, dataformats='HWC')
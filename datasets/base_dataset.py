from torch.utils.data import Dataset
import os
import glob
import torch
from typing import Any, Tuple, Dict
import numpy as np
import random
import hdf5storage
from matplotlib import pyplot as plt
import h5py

def readTXT(txt_path):
        with open(txt_path, 'r') as f:
            listInTXT = [line.strip() for line in f]

        return listInTXT

class BaseDataset(Dataset):
    def __init__(self, Path, split, crop_size, is_train=False, metamer=False, illu=False, cyclic_conv = False):
        self.GT_Path = Path + 'raw_pad/'
        self.GT_2xds_Path = Path + '2xds_pad/'
        self.illu = illu
        if split == 'Val':
            self.Test_GT_Path = Path + 'raw/Val/'
        else:
            self.Test_GT_Path = Path + 'raw/Val/'
        self.split = split
        print(split, metamer, illu)
        self.cyclic_conv = cyclic_conv
        if self.cyclic_conv:
            self.crop_size = crop_size
        else:
            self.crop_size = list(map(lambda x: x + 40,crop_size))
        self.is_train = is_train
        if self.is_train:
            if metamer == False:
                self.metamer = False
            else:
                self.metamer = True
                self.metamer_std = metamer[1] - metamer[0]
                self.metamer_bias = metamer[0]
        else:
            if metamer == False:
                self.metamer = 1
            else:
                self.metamer = float(metamer)

        if self.split == 'Train':
            self.mat_list = list(sorted(glob.iglob(self.GT_Path + split + "/*.mat")))
        elif self.split == 'Val':
            self.mat_list = list(sorted(glob.iglob(self.Test_GT_Path + "/*.mat")))
        elif self.split == 'Test':
            self.mat_list = list(sorted(glob.iglob(self.Test_GT_Path + "/*.mat")))

        self.size_list = readTXT(Path + 'size.txt')
        mat = hdf5storage.loadmat("optics/CMV4000_QE_norm.mat")
        Q = torch.tensor(mat['omega'].astype(np.float32))
        Q_T = Q.transpose(1, 0) # 3, N
        QQ_inv = torch.linalg.inv(torch.matmul(Q_T, Q))
        T = torch.matmul(Q, torch.matmul(QQ_inv, Q_T))
        self.T = T
        if self.is_train:
            self.pad = torch.nn.ReflectionPad2d((20, 20, 20, 20))
            self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return len(self.mat_list)
    
    def __getitem__(self, index):
        id = self.mat_list[index].split('/')[-1]
        if self.is_train:
            valid = False
            count = 0
            gt_path, max_H, max_W = self.random_downsample(index)
            while not valid:
                random_crop_h = random.randint(0, max_H)
                random_crop_w = random.randint(0, max_W)
                with h5py.File(gt_path + id, 'r') as f:
                    data = f['HSI'][random_crop_w:random_crop_w+self.crop_size[1],random_crop_h:random_crop_h+self.crop_size[0],:]
                spectrum = torch.tensor(data.astype(np.float32)).permute(2, 1, 0)
                if self.cyclic_conv:
                    grad = self.compute_image_gradients(spectrum[:, 30:-30, 30:-30])
                    grad_rel = grad / spectrum[:, 30:-30, 30:-30].mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
                else:
                    grad = self.compute_image_gradients(spectrum[:, 50:-50, 50:-50])
                    grad_rel = grad / spectrum[:, 50:-50, 50:-50].mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
                grad_max = grad.max()
                grad_rel_max = grad_rel.max()
                # valid = True
                if grad_max < 0.25 and grad_rel_max < 4 and count < 10:
                    pass
                else:
                    valid = True
            if count == 10:
                print(id)
            
            if self.illu:
                spectrum = self.real_illuminant(spectrum)
            if self.metamer:
                spectrum = self.random_metamer(spectrum, 1)
            spectrum = self.random_contrast(spectrum)
            spectrum = self.random_flip(spectrum)
            return {'id': id, 'spectrum': spectrum}
        else:
            gt_path = self.Test_GT_Path + '/'
            spectrum = torch.tensor(hdf5storage.loadmat(gt_path + id)['HSI'].astype(np.float32))
            spectrum = self.center_crop(spectrum, self.crop_size)
            if self.illu:
                spectrum = self.real_illuminant(spectrum, 0)
            spectrum = self.fix_metamer(spectrum, self.metamer)
            return {'id': id, 'spectrum': spectrum}
    
    def random_downsample(self, index):
        H, W = self.size_list[index].split(' ')
        p = random.random() 
        if p < 0.5:
            gt_path = self.GT_Path + self.split + '/'
            max_H = int(H) + 40 - self.crop_size[0]
            max_W = int(W) + 40 - self.crop_size[1]
        else:
            gt_path = self.GT_2xds_Path + self.split + '/'
            max_H = int(H) // 2 + 40 - self.crop_size[0]
            max_W = int(W) // 2 + 40 - self.crop_size[1]
        return gt_path, max_H, max_W

    def random_flip(self, data):
        p = random.random() 
        if p >= 0.75:
            data = data.flip(-1, -2)
        elif p >= 0.5:
            data = data.flip(-1)
        elif p >= 0.25:
            data = data.flip(-1)
        else:
            data = data

        return data
    
    def random_crop(self, data, crop_size):
        _, H, W = data.shape
        random_crop_h = random.randint(0, H - crop_size[0])
        random_crop_w = random.randint(0, W - crop_size[1])
        data = data[:, random_crop_h:random_crop_h+crop_size[0], random_crop_w:random_crop_w+crop_size[1]]
        return data
    
    def center_crop(self, data, crop_size):
        _, H, W = data.shape
        center_crop_h = (H - crop_size[0])//2
        center_crop_w = (W - crop_size[1])//2
        data = data[:, center_crop_h:center_crop_h+crop_size[0], center_crop_w:center_crop_w+crop_size[1]]
        return data
    
    def random_contrast(self, data, min=-0.2, max=0):
        random_contrast = random.uniform(min, max) + 1.0
        data = random_contrast * data
        return data
    
    def random_metamer(self, spectrum, n):
        spectrum = spectrum.permute(1, 2, 0).unsqueeze(-1)
        SS = torch.matmul(self.T, spectrum)
        B = spectrum - SS
        alpha = torch.rand(1,1,1,n) * self.metamer_std + self.metamer_bias
        new_S = SS + alpha * B
        new_S = new_S.clip(0, None).permute(3, 2, 0, 1)
        return new_S
    
    def fix_metamer(self, spectrum, alpha):
        spectrum = spectrum.permute(1, 2, 0).unsqueeze(-1)
        if alpha == 1:
            new_S = spectrum
        else:
            SS = torch.matmul(self.T, spectrum)
            B = spectrum - SS
            new_S = SS + alpha * B
        new_S = new_S.clip(0, None).permute(3, 2, 0, 1)
        return new_S

    def compute_image_gradients(self, image):
        if image.ndim == 3: 
            image = image.unsqueeze(1)
        gradient_x = torch.nn.functional.conv2d(image, self.sobel_x, padding=1)
        gradient_y = torch.nn.functional.conv2d(image, self.sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        
        return gradient_magnitude.squeeze()[:,1:-1,1:-1]
    
    def real_illuminant(self, spectrum, p = None):
        if p is None:
            p = random.random()
        if p < 0.5:
            weight = [
                0.2600, 0.3151, 0.3661, 0.4196, 0.4515, 0.5643, 0.5984, 0.6272, 0.6888, 0.7973, 0.8180, 0.8680, 0.9079, 0.9441, 1.0497, 1.1190, 1.1836, 1.2313, 1.3366, 1.3885, 1.4452, 1.5004, 1.5218, 1.53, 1.52, 1.4928, 1.4702, 1.4287, 1.3484, 1.2382, 1.0563, 1.0904, 1.1148, 0.9673, 0.8360, 0.8128, 0.7914, 0.7865, 0.7733, 0.7654, 0.7549, 0.7311, 0.3664
            ]
            weight = torch.tensor(weight, dtype=torch.float32)
            weight = weight / weight.mean()
            return spectrum * weight.view(-1, 1, 1)
        else:
            return spectrum
        
    def halogen_illuminant(self, spectrum):
        p = random.random()
        if p < 0.5:
            weight = [
                0.457856249825341, 0.494468747404492, 0.531285699978910, 0.568126768282237,
                0.604820405029097, 0.641205149049860, 0.677130596331610, 0.712458078197391,
                0.747061078713516, 0.780825423878926, 0.813649274546686, 0.845442953636897,
                0.876128636259864, 0.905639929073161, 0.933921362703062, 0.960927818492292,
                0.986623908285331, 1.01098332349842, 1.03398816739193, 1.05562828229939,
                1.07590058158921, 1.09480839434930, 1.11236082919317, 1.12857216218279,
                1.14346125263973, 1.15705098956046, 1.16936777045051, 1.18044101363183,
                1.19030270444379, 1.19898697523680, 1.20652971863522, 1.21296823321095,
                1.21834090044823, 1.22268689168443, 1.22604590357023, 1.22845792049748,
                1.22996300238681, 1.23060109620237, 1.23041186956250, 1.22943456483754,
                1.22770787216478, 1.22526981986188, 1.22215768078159
            ]
            weight = torch.tensor(weight, dtype=torch.float32)
            return spectrum * weight.view(-1, 1, 1)
        else:
            return spectrum
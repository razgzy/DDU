from typing import Tuple
import lightning as L
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from datasets import ICVLDataset, VNIRSRDataset
Name_Class = {'icvl': ICVLDataset, 
              'vnirsr': VNIRSRDataset,
              }

class SpectrumDataset(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, crop_size: Tuple = (128, 128), val_crop_size: Tuple = (512, 512), test_crop_size: Tuple = (512, 512), batch_size: int = 4, num_workers: int = 4, aug_metamer = (-2, 2), aug_illu = False, accumulate_grad_batches: int = 1, cyclic_conv: bool = False, step_per_epoch: int = 5000):
        super().__init__()
        self.crop_size = crop_size
        self.val_crop_size = val_crop_size
        self.test_crop_size = test_crop_size
        self.step_per_epoch = step_per_epoch
        self.num_samples = self.step_per_epoch * batch_size
        self.batch_size = batch_size // accumulate_grad_batches
        self.num_workers = num_workers // accumulate_grad_batches
        self.aug_metamer = aug_metamer
        self.aug_illu = aug_illu
        self.cyclic_conv = cyclic_conv
        self.train_dataset_classes = []
        self.train_dataset_name = train_dataset
        for x in self.train_dataset_name:
            self.train_dataset_classes.append(Name_Class[x])
        self.val_dataset_name = val_dataset
        self.val_dataset_classes = []
        for x in self.val_dataset_name:
            self.val_dataset_classes.append(Name_Class[x.split('_')[0]])
        self.test_dataset_name = test_dataset
        self.test_dataset_classes = []
        for x in self.test_dataset_name:
            self.test_dataset_classes.append(Name_Class[x.split('_')[0]])

    def setup(self, stage):
        if stage == 'fit':
            self.train_datasets = []
            for i, dataset in enumerate(self.train_dataset_classes):
                self.train_datasets.append(dataset('Train', self.crop_size, is_train=True, metamer=self.aug_metamer, illu = self.aug_illu, cyclic_conv=self.cyclic_conv))
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_datasets = []
            for i, dataset in enumerate(self.val_dataset_classes):
                self.val_datasets.append(dataset('Val', self.val_crop_size, is_train=False, metamer=self.val_dataset_name[i].split('_')[1], illu=('illu' in self.val_dataset_name[i]), cyclic_conv=self.cyclic_conv))
        elif stage == 'validate':
            self.val_datasets = []
            for i, dataset in enumerate(self.val_dataset_classes):
                self.val_datasets.append(dataset('Val', self.val_crop_size, is_train=False, metamer=self.val_dataset_name[i].split('_')[1], illu=('illu' in self.val_dataset_name[i]), cyclic_conv=self.cyclic_conv))
        elif stage == 'test':
            self.test_datasets = []
            for i, dataset in enumerate(self.test_dataset_classes):
                self.test_datasets.append(dataset('Test', self.test_crop_size, is_train=False, metamer=self.test_dataset_name[i].split('_')[1], illu=('illu' in self.test_dataset_name[i]), cyclic_conv=self.cyclic_conv))
        
    def train_dataloader(self):
        num_samples = self.num_samples
        sampler = RandomSampler(self.train_dataset, replacement=False, num_samples=num_samples)
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=2)
        return dataloader
    
    def val_dataloader(self):
        dataloaders = []
        batch_size = self.batch_size//2
        num_workers=self.num_workers//2
        if batch_size == 0:
            batch_size = 1
            num_workers = 2
        for dataset in self.val_datasets:
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True))
        return dataloaders
    
    def test_dataloader(self):
        dataloaders = []
        for dataset in self.test_datasets:
            dataloaders.append(DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True))
        return dataloaders
    
    # def state_dict(self):
    #     # track whatever you want here
    #     state = {"current_train_batch_index": self.current_train_batch_index}
    #     return state

    # def load_state_dict(self, state_dict):
    #     # restore the state based on what you tracked in (def state_dict)
    #     self.current_train_batch_index = state_dict["current_train_batch_index"]
from .base_dataset import BaseDataset

Path = '/data3/HyperSpectral/XHSI480900/'

class VNIRSRDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(Path, split, *args, **kwargs)
        print('XHSI ' + split, len(self.mat_list))


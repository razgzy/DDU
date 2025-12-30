from .base_dataset import BaseDataset

Path = '/data3/HyperSpectral/VNIRSR480900/'

class VNIRSRDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(Path, split, *args, **kwargs)
        print('VNIRSR ' + split, len(self.mat_list))


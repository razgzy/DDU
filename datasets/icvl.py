from .base_dataset import BaseDataset

Path = '/data2/HyperSpectral/ICVL480900/'

class ICVLDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(Path, split, *args, **kwargs)
        print('ICVL ' + split, len(self.mat_list))


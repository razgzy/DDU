import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import time
import lightning as L
from lightning.pytorch.cli import LightningCLI
import torch
torch.set_printoptions(precision=6)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from lightning_modules.Trainer import customTrainer

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")
        parser.link_arguments("trainer.accumulate_grad_batches",  "model.init_args.accumulate_grad_batches")
        parser.link_arguments("model.init_args.cyclic_conv",  "data.init_args.cyclic_conv")
        parser.link_arguments("data.init_args.crop_size", "model.init_args.crop_size")
        parser.link_arguments("data.init_args.val_crop_size",  "model.init_args.val_crop_size")
        parser.link_arguments("data.init_args.test_crop_size",  "model.init_args.test_crop_size")

def cli_main():
    cli = MyLightningCLI(trainer_class=customTrainer, parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == '__main__':
    cli_main()

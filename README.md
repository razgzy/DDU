# A Deep Unfolding Framework for Diffractive Snapshot Spectral Imaging

## Environment

Pytorch 2.7.1+cu118 (> 2.0)
lightning == 2.6.0.dev0 (> 2.0)
omegaconf
jsonargparse[signatures]

## Data

Download preprocessed ICVL ([Quark](https://pan.quark.cn/s/7f2eafaf7e14), Code: 5UXz) and VNIRSR ([Quark](https://pan.quark.cn/s/b53f7674570d), Code: CfF9) datasets and modify the path in datasets/icvl.py and datasets/vnirsr.py

## Run

Train with 
```bash
   ./train.sh configs/trainv2.yaml
```
Test with
```bash
   ./test.sh configs/testv2.yaml
```



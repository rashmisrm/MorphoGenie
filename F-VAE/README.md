# FactorVAE
Pytorch implementation of FactorVAE proposed in Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])
<br>

### Dependencies
```
python 3.6.4
pytorch 0.4.0 (or check pytorch-0.3.1 branch for pytorch 0.3.1)
visdom
tqdm
```
<br>

```
Preprocess steps include Centring the cells followed by cropping and resizing.

### Usage
initialize visdom

```
python -m visdom.server
```
you can reproduce results below as follows
```
```
```
e.g.
python main.py --name run_celeba --dataset ./data/ProcesssedData/CellCycle-QPI --gamma 6.4 --lr_VAE 1e-4 --lr_D 5e-5 --z_dim 10 ...
```
check training process on the visdom server
```
localhost:8097

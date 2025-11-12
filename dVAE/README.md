# Train a VAE to learn Disentangled Latent Representations 

This is a Pytorch implementation of FactorVAE proposed in Disentangling by Factorising, Kim et al.([http://arxiv.org/abs/1802.05983])
<br>

### Dependencies

Create a new conda envirinment and install pytorch 

```
conda create --name dVAE-Env 
conda activate dVAE-Env

```

```
pip install visdom tqdm matplotlib seaborn 
```
<br>

Preprocess steps include Centering the cells followed by cropping and resizing.

### Usage

Initialize visdom. 

```
python -m visdom.server
```
Train the VAE from dVAE directory using 
```
e.g.
python main.py --name dVAE_Model1 --dataset ./ProcesssedData/LC --gamma 10 --lr_VAE 1e-4 --lr_D 5e-5 --z_dim 10 

```

Check training process on the visdom server

```
localhost:8097

```

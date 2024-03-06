# MorphoGenie
#Testing
Is an integrative deep-learning framework for single-cell morphological profiling.

MorphoGenie uses neural network combination of VAE and GAN as porposed in model in ID- GAN (https://arxiv.org/abs/2001.04296)

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Fig-1B.png)

## Requirements

- Pytorch,
- Tensorboard,
- Tested on Windows

Setting up testing environment

Install the pytorch cuda version

conda create —name MorphoGenie python=3.8.10

conda install -c anaconda tensorboard

conda install -c anaconda matplotlib

conda install -c anaconda scikit-learn

## Folder structure

Parent Folder

Label1

Folder

1.jpg

## Image Preprocessing

The single-cell images in the dataset to be tested need to be segmented, cropped and centered.

## Testing with pre-trained models

```

Load the model to test with the desired dataset

- Predicted latent factors for downstream analysis

- Image reconstructions for interpreting disentangled latent space

```

train\_test.py, args='--config cells\_650.yaml --dvae\_name \<VAE\_Model\_Name\> --name \<GAN\_Model\_Name\>', wdir='../MorphoGenie')

## Pre-processing

## Lung Cancer Dataset

train\_test.py, args='--config cells\_650.yaml --dvae\_name VAE\_LC --name GAN\_LC', wdir='../MorphoGenie')

## Cell Painting Assay

train\_test.py, args='--config cells\_650.yaml --dvae\_name VAE\_CPA --name GAN\_CPA', wdir='../MorphoGenie')

## CellCycle

train\_test.py, args='--config cells\_650.yaml --dvae\_name cells\_trial3 --name GAN\_Trial3', wdir='../MorphoGenie')

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Animation.png)

## Epithelial to Mesenchymal transition

train\_test.py, args='--config cells\_650.yaml --dvae\_name cells\_trial3 --name GAN\_Trial3', wdir='../MorphoGenie')

## Test Results

### Downstream

Downstream analysis is performed by visualizing the disentagled latent space in 2 dimensions using dimensionality reduction technique called UMAP Phate

#### Lung Cancer

#### CPA

#### CellCycle

For datasets the reveal continuous progressions, or develop into pathways, MorphoGenie uses VIA to perform trajectory inference.
![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/AnimateCCy.png)

#### EMT
![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Animate.png)


#### Interpreting Disentangled Latent Space in MorphoGenie

Images of the traversals are generated by setting flag saveTraversals=True and the traversal images are saved in the folder \<outputs\>

#### Train models with new datasets

- Step 1: Train VAEs.

```

python dvae\_main.py --dataset [dataset\_name] --name [dvae\_run\_name] --c\_dim [c\_dim] --beta [beta]

```

, where `[dataset_name]` can be one of `LC`, `CellCycle`, `CellPainingAssay`, and `EMT`.

please refer to `dvae_main.py` for the details.

- Stage 2: Train ID-GAN through information distillation loss.

```

python train.py --config [config\_name] --dvae\_name [dvae\_run\_name] --name [idgan\_run\_name]

```

## Results

Results, including checkpoints, tensorboard logs, and images can be found in `outputs` directory.

## Acknowledgement

Pytorch implementation on "High-fidelity Synthesis with Disentangled Representation" (https://arxiv.org/abs/2001.04296). \<br\>

This code is built based on the following code repositories:

1. Information Distillation GAN (ID-GAN): https://github.com/1Konny/idgan.git

2. Factor-VAE: [https://github.com/1Konny/FactorVAE.git]
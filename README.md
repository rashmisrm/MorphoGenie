# MorphoGenie
#Testing
MorphoGenie, a new deep learning framework for unsupervised, interpretable single-cell morphological profiling and analysis. MorphoGenie standsout in three key attributes: 
	(1) High-fidelity Image Reconstruction: MorphoGenie utilizes a hybrid architecture that capitalizes on the unparalleled strengths of the variant of variational Autoencoders (VAEs) and generative Adversarial Networks (GANs) to achieve interpretable, high-quality cell image generation. 

	(2) Interpretability: MorphoGenie adopts a VAE-based method to learn a compact, interpretable, and transferable disentangled representation for single-cell morphological analysis. In contrast to the prior work, we propose a novel technique for interpreting the learned representation by extracting handcrafted features from reconstructed images produced by latent traversals, facilitating the discovery of biologically meaningful inferences, especially the heterogeneities of cell types and lineages. 

	(3) Generalizability: MorphoGenie is widely adaptable across various imaging modalities and experimental conditions, promoting cross-study comparisons and reusable morphological profiling results. The model generalizes to unseen single-cell datasets and different imaging modalities while providing explanations for its predictions. Overall, MorphoGenie could spearhead new strategies for conducting comprehensive morphological profiling and make biologically meaningful discoveries across a wide range of imaging modalities.


MorphoGenie uses neural network combination of VAE and GAN as porposed in model in ID-GAN 

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Fig-1B.png)

## Requirements
```
- Pytorch,
- Tensorboard,
- Tested on Windows
```
Setting up testing environment

Install the pytorch cuda version
```
conda create —name MorphoGenie python=3.8.10

conda install -c anaconda tensorboard

conda install -c anaconda matplotlib

conda install -c anaconda scikit-learn
```

Other requirements
```
spyder, pandas, matplotlib, seaborn, umap, tsne, phate, numpy
```
## Folder structure

Parent Folder

```
Label1

	Folder

		Image1.jpg
		Image2.jpg
		.
		.
		.
		ImageM.jpg

Label2

	Folder

		Image1.jpg
		Image2.jpg
		.
		.
		.
		ImageN.jpg
Label3

	Folder

		Image1.jpg
		Image2.jpg
		.
		.
		.
		ImageO.jpg
```
## Image Preprocessing

The single-cell images in the dataset to be tested are required to be segmented, cropped as a single cell image and centered to avoid biases in training due to positional dependence of the cells.


## Testing with pre-trained models

```

Load the model to test with the desired dataset

	- Moprhological Features Extraction: VAEfor downstream classification analysis and interpretation

	- Image reconstructions for interpreting disentangled latent space

	- Generalizability Tests : Use a pre-trained model to predict and perform downstream analysis based on the MorphoGenie features.



```

train\_test.py, args='--config cells\_650.yaml --dvae\_name \<VAE\_Model\_Name\> --name \<GAN\_Model\_Name\>', wdir='../MorphoGenie')

## Pre-processing



## Test Results

### Downstream

Downstream analysis is performed by visualizing the disentagled latent space in 2 dimensions using dimensionality reduction technique called UMAP Phate

### Interpretability
Traversal

### Generalizability

For datasets the reveal continuous progressions, or develop into pathways, MorphoGenie uses VIA to perform trajectory inference.


![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/AnimateCCy.gif)

#### EMT


![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Animate.gif)


#### Interpreting Disentangled Latent Space in MorphoGenie
```
Images of the traversals are generated by setting flag saveTraversals=True and the traversal images are saved in the folder \<outputs\>
```

#### Train models with new datasets

- Step 1: Train VAEs.

```

python dvae\_main.py --dataset [dataset\_name] --name [dvae\_run\_name] --c\_dim [c\_dim] --beta [beta]

```
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
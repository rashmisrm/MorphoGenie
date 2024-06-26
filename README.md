# MorphoGenie
MorphoGenie, a new deep learning framework for unsupervised, interpretable single-cell morphological profiling and analysis. MorphoGenie standsout in three key attributes: 
	(1) High-fidelity Image Reconstruction: MorphoGenie utilizes a hybrid architecture that capitalizes on the unparalleled strengths of the variant of variational Autoencoders (VAEs) and generative Adversarial Networks (GANs) to achieve interpretable, high-quality cell image generation. 
	(2) Interpretability: MorphoGenie adopts a VAE-based method to learn a compact, interpretable, and transferable disentangled representation for single-cell morphological analysis. In contrast to the prior work, we propose a novel technique for interpreting the learned representation by extracting handcrafted features from reconstructed images produced by latent traversals, facilitating the discovery of biologically meaningful inferences, especially the heterogeneities of cell types and lineages. 
	(3) Generalizability: MorphoGenie is widely adaptable across various imaging modalities and experimental conditions, promoting cross-study comparisons and reusable morphological profiling results. The model generalizes to unseen single-cell datasets and different imaging modalities while providing explanations for its predictions. Overall, MorphoGenie could spearhead new strategies for conducting comprehensive morphological profiling and make biologically meaningful discoveries across a wide range of imaging modalities.


MorphoGenie uses neural network combination of VAE and GAN as porposed in model in ID-GAN 

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Intro.png)

## Requirements
```
- Pytorch,
- Tensorboard,
- spyder, pandas, matplotlib, seaborn, umap, tsne, phate, numpy
Tested on windows
```
## Setting up testing environment

Install the pytorch cuda version

```
conda create —name MorphoGenie python=3.8.10

conda activate MorphoGenie

conda install -c anaconda tensorboard

conda install -c anaconda matplotlib

conda install -c anaconda scikit-learn
```

## Folder structure

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

```

## Image Preprocessing

The single-cell images in the dataset to be tested are required to be segmented, cropped as a single cell image and centered to avoid biases in training due to positional dependence of the cells. 


## Testing with pre-trained models
Load the pre-trained model and select the dataset for testing. This step generates Latent.csv and Label.csv for downstream analysis such as cell data visualization, classification and interpretation tasks.

```
python MorphoGenie_test.py, args='--config cells\_650.yaml --dvae\_name \<VAE\_Model\_Name\> --name \<GAN\_Model\_Name\>', wdir='../MorphoGenie')
```

### Downstream Visualization

Downstream analysis is performed by visualizing the disentagled latent space in 2 dimensions using dimensionality reduction technique called UMAP Phate

For datasets the reveal continuous progressions, or develop into pathways, MorphoGenie uses VIA to perform trajectory inference.



![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/AnimateCCy.gif)


![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Animate.gif)


### Interpreting Disentangled Latent Space in MorphoGenie

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Disent.png)

MorphoGenie's interpretability is enhanced through analysis of how its disentangled latent space relates to the physical characteristics of individual cells. These characteristics are identified through a hierarchical analysis that categorizes features into a structured framework, ranging from subtle textures to more distinct properties like cell size, shape, and density. Using this, MorphoGenie creates a profile called "Interpretation Heatmap" that enables meaningful and biologically relevant interpretations of the disentangled representations.


To interpret MorphoGenie's dientangled latent space, traversal reconstructions are saved by setting the flag to True.

```
python MorphoGenie_test.py, args='--config cells\_650.yaml --dvae\_name \<VAE\_Model\_Name\> --name \<GAN\_Model\_Name\>', wdir='../MorphoGenie')
```

Images of the traversals are generated by setting flag SaveTraversals=True and the traversal images are saved in the folder \<outputs\>


### Generalizability

To assess MorphoGenie's generalizability, model pre-trained on a dataset from one imaging modality can be employed to test its performance on unseen datasets with different image contrasts. MorphoGenie could apply its trained latent representations to perform accurate downstream analyses and predictions on these new test datasets, without any retraining.


To test generalizability, the pre-trained model is loaded and latent features are predicted by simply inputting the location of a new segmented and pre-processed dataset from an entirely new imaging modality.


### Train models with new datasets

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Idgan.png)

- Step 1: Train VAEs.

```

python dvae\_main.py --dataset [dataset\_name] --name [dvae\_run\_name] --c\_dim [c\_dim] --beta [beta]

```
Where `[dataset_name]` can be one of `LC`, `CellCycle`, `CellPainingAssay`, and `EMT`.
please refer to `dvae_main.py` for the details.

- Step 2: Train ID-GAN through information distillation loss.


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
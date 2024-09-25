# MorphoGenie - Unsupervised Deep Learning Framework for GENeralizable, Interpretable/Explainable Single-Cell Morphological Profiling

The intersection of advanced microscopy and machine learning is transforming the field of cell biology, enabling a more quantitative and data-driven approach. Traditional methods of morphological profiling, which rely on manual feature extraction, are often time-consuming, labor-intensive, and susceptible to human bias. Deep learning offers a promising alternative, but its effectiveness is hindered by its "black-box" operation and its dependence on extensive labeled data. 

MorphoGenie addresses these challenges by introducing an unsupervised deep-learning framework for single-cell morphological profiling. This innovative tool generates high-fidelity image reconstructions, enabling disentangled representation learning and a compact, interpretable latent space. This allows for the extraction of biologically meaningful features without human annotation, overcoming the "curse of dimensionality" inherent in manual methods.

MorphoGenie standsout in three key attributes: 
1. High-fidelity Image Reconstruction: MorphoGenie utilizes a hybrid architecture that capitalizes on the unparalleled strengths of the variant of variational Autoencoders (VAEs) and generative Adversarial Networks (GANs) to achieve interpretable, high-quality cell image generation. 
2. Interpretability: MorphoGenie adopts a VAE-based method to learn a compact, interpretable, and transferable disentangled representation for single-cell morphological analysis. In contrast to the prior work, we propose a novel technique for interpreting the learned representation by extracting handcrafted features from reconstructed images produced by latent traversals, facilitating the discovery of biologically meaningful inferences, especially the heterogeneities of cell types and lineages. 
3. Generalizability: MorphoGenie is widely adaptable across various imaging modalities and experimental conditions, promoting cross-study comparisons and reusable morphological profiling results. The model generalizes to unseen single-cell datasets and different imaging modalities while providing explanations for its predictions. Overall, MorphoGenie could spearhead new strategies for conducting comprehensive morphological profiling and make biologically meaningful discoveries across a wide range of imaging modalities.



![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Intro.png)

## Requirements
```
- Pytorch,
- Tensorboard,
- pandas, matplotlib, seaborn, umap, numpy, tqdm
Tested on windows
```
## Setting up testing environment

Install the pytorch cuda version suitable for the OS in MorphoGenie environment [Link](https://pytorch.org/get-started/locally/). 

```
conda create —name MorphoGenie python=3.8.10

conda activate MorphoGenie

conda install -c anaconda pandas=1.4.2 matplotlib=3.5.1 seaborn=0.11.2 umap-learn numpy=1.22 tqdm=4.63.0 scikit-image umap-learn scikit-learn=1.0.2 pillow=9.2.0

conda install -c anaconda tensorboard

```

### Dataset

The processed datasets for testing are available [here.](https://hkuhk-my.sharepoint.com/:f:/g/personal/rashmism_hku_hk/El4Ew1HJP5pGgThVYwuaN6kB9cXScl89KL0RSCxRPQr-vg?e=p2cgTt)

| Dataset       | Folder Name| Imaging Modality |
| ------------- | -----------|------------------|
| Lung Cancer   | LC         | QPI              |
| Cell Painting Assay| CPA   | Fluorescence     |
| Epithelial to Mesencymal Transition| EMT| Fluorescence|
| CellCycle     | CCy        | QPI|

### Folder structure

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

### Image Preprocessing

The single-cell images in the dataset to be tested are required to be segmented, cropped as a single cell image. Segmentation is performed using any of the available tools such as Cellpose.


## Testing with pre-trained models
Load the [pre-trained model](https://hkuhk-my.sharepoint.com/:f:/g/personal/rashmism_hku_hk/EnFvx47idy1MpOjuIdEdYzAB50xStrq6XqEt00ZKqHrC0Q?e=Shah2a) and select the dataset for testing. This step generates Latent.csv and Label.csv for downstream analysis such as cell data visualization, classification and interpretation tasks. 

```
python MorphoGenie_Test.py --config cells_650.yaml --VAE_name LC_VAE --GAN_name LC_GAN  --Traversal=True --dataset=LC
```

Alternatively, MorphoGenie_Test.ipynb can be employed for testing the data.


## Generalizability

To assess MorphoGenie's generalizability, model pre-trained on a dataset from one imaging modality can be employed to test its performance on unseen datasets with different image contrasts. MorphoGenie could apply its trained latent representations to perform accurate downstream analyses and predictions on these new test datasets, without any retraining.


To test generalizability, the pre-trained model is loaded and latent features are predicted by simply inputting the location of a new segmented and pre-processed dataset from an entirely new imaging modality.

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Generalizability.png)



## Interpreting Disentangled Latent Space in MorphoGenie



MorphoGenie's interpretability is enhanced through analysis of how its disentangled latent space relates to the physical characteristics of individual cells. These characteristics are identified through a hierarchical analysis that categorizes features into a structured framework, ranging from subtle textures to more distinct properties like cell size, shape, and density. Using this, MorphoGenie creates a profile called "Interpretation Heatmap" that enables meaningful and biologically relevant interpretations of the disentangled representations.

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Disent.png)

Traversal reconstructions are generated (setting flag Traverse = True) to interpret MorphoGenie's disentangled latent space. The process involves:
1.⁠ ⁠50 traversal sets
2.⁠ ⁠Variance matrix computation
3.⁠ ⁠Averaging for heatmap generation


The resulting heatmap reveals latent space structure and feature correlations

```
python MorphoGenie_Test.py --config cells_650.yaml --VAE_name LC_VAE --GAN_name LC_GAN  --Traversal=True --dataset=LC
```

Images of the traversals are generated by setting flag SaveTraversals=True and the traversal images are saved in the folder \<outputs\>


## Train models with new datasets

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Idgan.png)

- Step 1: Train VAEs. This step requires setting up an different enviromnent to train the VAE according to: [Factor-VAE](https://github.com/1Konny/FactorVAE.git)

```

python dvae\_main.py --dataset [dataset\_name] --name [dvae\_run\_name] --c\_dim [c\_dim] --beta [beta]

```
Where `[dataset_name]` can be one of `LC`, `CellCycle`, `CellPainingAssay`, and `EMT`.
please refer to `dvae_main.py` for the details.

- Step 2: Train ID-GAN through information distillation loss.


```
python train.py --config [config\_name] --dvae\_name [dvae\_run\_name] --name [idgan\_run\_name]

```

### Saving Results

Results, including checkpoints, tensorboard logs, and images can be found in `outputs` directory.

## Acknowledgement

Pytorch implementation on "High-fidelity Synthesis with Disentangled Representation" (https://arxiv.org/abs/2001.04296)

This code is built based on the following code repositories:

1. Information Distillation GAN (ID-GAN): https://github.com/1Konny/idgan.git

2. Factor-VAE: [https://github.com/1Konny/FactorVAE.git]


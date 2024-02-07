
# MorphoGenie
Is an integrative deep-learning framework for single-cell morphological profiling.




## Requirements

## Setting up testing environment
###	Install the pytorch cuda version
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

	Load the model to test with the desired dataset
		- Predicted latent factors for downstream analysis
		- Image reconstructions for interpreting disentangled latent space


	train_test.py, args='--config cells_650.yaml --dvae_name <VAE_Model_Name> --name <GAN_Model_Name>', wdir='../MorphoGenie')

## Pre-processing


## Lung Cancer Dataset

	train_test.py, args='--config cells_650.yaml --dvae_name VAE_LC --name GAN_LC', wdir='../MorphoGenie')

## Cell Painting Assay 
	train_test.py, args='--config cells_650.yaml --dvae_name VAE_CPA --name GAN_CPA', wdir='../MorphoGenie')

## CellCycle 

	train_test.py, args='--config cells_650.yaml --dvae_name cells_trial3 --name GAN_Trial3', wdir='../MorphoGenie')

## Epithelial to Mesenchymal transition
train_test.py, args='--config cells_650.yaml --dvae_name cells_trial3 --name GAN_Trial3', wdir='../MorphoGenie')

# Test Results

#Train MorphoGenie in two steps


- Step 1: Train VAEs.
```
python dvae_main.py --dataset [dataset_name] --name [dvae_run_name] --c_dim [c_dim] --beta [beta]
```
, where `[dataset_name]` can be one of `dsprites`, `celeba`, `cars`, and `chairs`.
please refer to `dvae_main.py` for the details.

- Stage 2: Train ID-GAN through information distillation loss.
```
python train.py --config [config_name] --dvae_name [dvae_run_name] --name [idgan_run_name]
```


## Results
Results, including checkpoints, tensorboard logs, and images can be found in `outputs` directory.


# Acknowledgement

## ID-GAN
Pytorch implementation on "High-fidelity Synthesis with Disentangled Representation" (https://arxiv.org/abs/2001.04296). <br>
For ID-GAN augmented with Variational Discriminator Bottleneck (VDB) or VGAN, please refer to the vgan [branch](https://github.com/1Konny/idgan/tree/vga).


This code is built on the repos as follows:
1. Beta-VAE: [https://www.github.com/1Konny](https://www.github.com/1Konny)
2. GAN with R2 regularization: [https://github.com/LMescheder/GAN_stability](https://github.com/LMescheder/GAN_stability)
3. VGAN: [https://github.com/akanazawa/vgan](https://github.com/akanazawa/vgan) 

# Citation
If you find our work useful for your research, please cite our paper.
```
@article{lee2020highfidelity, 
    title={High-Fidelity Synthesis with Disentangled Representation}, 
    author={Wonkwang Lee and Donggyun Kim and Seunghoon Hong and Honglak Lee}, 
    year={2020}, 
    journal={arXiv preprint arXiv:2001.04296}, 
}
```


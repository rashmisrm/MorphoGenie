# Image Pre-processing 

## Image preprocessing pipeline involves several key steps to prepare images for analysis:

1. #### Segmentation:

     #### a. Single-cell images captured (imaging flow cytometer): 
	Intensity threshold-based segmentation is used (Lung Cancer, Cell Cycle datasets). 

     #### b. Multiple cells in the field of view (cell culture plates): 
	Cellpose (Cellpose v2.1.1) is utilized for batch processing of image segmentation, specifically leveraging the 'cyto2' model for segmenting images  (Cell Painting dataset and EMT). 

2. #### Background noise removal: 
	Noise is removed while preserving cell body information.

3. #### Center cropping and resizing: 
	Images are center-cropped and resized to 256 x 256 pixels.

4. #### Cell alignment: 
	Cells are aligned to the center of the image frame to prevent positional features from influencing analysis.


### The code for these steps is organized in individual folders based on segmentation requirements.

1. Download and extract the raw data files: [Link](https://hkuhk-my.sharepoint.com/my?id=%2Fpersonal%2Frashmism%5Fhku%5Fhk%2FDocuments%2FMorphoGenieLink%2FRawDatasets&csf=1&web=1&CID=ee3d57ad%2D949c%2D4516%2D85d6%2D3257a6467d3b&FolderCTID=0x0120007AB2941E62AA1B49B2FF62BCAA405A51). 

2. Copy relevant files: Place the extracted raw data files in the respective dataset folder.

3. Run Pre-precossing steps 

	a. For single cell images (Lung Cancer, CellCycle): Segmentation followed by other preprocessing steps are included in the '''CropSeg.m'''.
	 Run the CropSeg.m from the directory specific to the dataset you are performing preprocessing. This step saves singlecell images in a folder named 	'''Cropped''' in the current directory.

 	b. For multiple cell images:
		
		i. Use batch processing code to generate masks using CellPose batch processing code '''CellPose_Batch.ipynb'''
		Details pertaining to using CellPose can be found here [Link](https://cellpose.readthedocs.io/en/latest/index.html)

		ii. Use the Matlab code '''CropSeg.m''' to save the cropped images in the *Cropped* folder in the working directory
		 - Single Channel such as EMT saves the 
		 - Multiple channels datasets (Cell Painting): Processed Images are saved in 5 different folders corresponding to 5 channels.
		 - Each folder contains images of chnell names after a unique cell id. For example. Cell #1 whose organells are arranged in 5 channel is named as 1.png in all 5 channel folders.  


e. Use matlab code to create individual cell images, centered and noise free provided separately for datasets.




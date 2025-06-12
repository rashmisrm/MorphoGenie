## Image Pre-processing 

### Image preprocessing pipeline involves several key steps to prepare images for analysis:

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

### Code Structure

The code is organized in individual folders based on segmentation requirements. Each folder contains the necessary scripts for pre-processing specific datasets.

### Prerequisites
	MATLAB, CellPose

### Set-up

1. Download and extract the raw data files: [Link](https://hkuhk-my.sharepoint.com/my?id=%2Fpersonal%2Frashmism_hku_hk%2FDocuments%2FMorphoGenieLink%2FRawDatasets&csf=1&web=1&CID=ee3d57ad-949c-4516-85d6-3257a6467d3b&FolderCTID=0x0120007AB2941E62AA1B49B2FF62BCAA405A51)
2. Place the extracted raw data files in the respective dataset folder.


### Usage

*Single-Cell Images*
1. Run `CropSeg.m` from the directory specific to the dataset.
2. The pre-processed images will be saved in a folder named `Cropped` in the current directory.

*Multiple Cell Images*
1. Use `CellPose_Batch.ipynb` for batch processing and generating masks. [Link](https://cellpose.readthedocs.io/en/latest/index.html)
2. Run `CropSeg.m` to save cropped single-cell images in the `Cropped` folder.

- Single-Channel Datasets (e.g., EMT): Processed images are saved in a single folder.
- Multi-Channel Datasets (e.g., Cell Painting): Processed images are saved in separate folders for each channel (5 folders in total). 
- Each folder contains images named after a unique cell ID, with the same ID used across all channels. For example, Cell #1's images would be named `1.png` in each of the 5 channel folders.


To run the MorphoGenie test script on processed images:

1. Place the `Cropped` folder in the `ProcessedData` folder located in the root directory.
2. Ensure the files are organized according to the specified folder structure.






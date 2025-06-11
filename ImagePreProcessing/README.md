# Image Pre-processing 

## Image preprocessing pipeline involves several key steps to prepare images for analysis:

1. ### Segmentation:

     #### a. Single-cell images captured (imaging flow cytometer): 
	Intensity threshold-based segmentation is used (Lung Cancer, Cell Cycle datasets). 

     #### b. Multiple cells in the field of view (cell culture plates): 
	Cellpose (Cellpose v2.1.1) is utilized for batch processing of image segmentation, specifically leveraging the 'cyto2' model for segmenting images  (Cell Painting dataset and EMT). 

2. #### Background noise removal: Noise is removed while preserving cell body information.

3. #### Center cropping and resizing: Images are center-cropped and resized to 256 x 256 pixels.

4. #### Cell alignment: Cells are aligned to the center of the image frame to prevent positional features from influencing analysis.


### The code for these steps is organized in individual folders based on segmentation requirements.

a. _Download and extract the raw data files_: [Link](https://hkuhk-my.sharepoint.com/my?id=%2Fpersonal%2Frashmism%5Fhku%5Fhk%2FDocuments%2FMorphoGenieLink%2FRawDatasets&csf=1&web=1&CID=ee3d57ad%2D949c%2D4516%2D85d6%2D3257a6467d3b&FolderCTID=0x0120007AB2941E62AA1B49B2FF62BCAA405A51). 

b. _Copy relevant files_: Place the extracted raw data files in the respective dataset folder.

c. Use batch processing code to extract masks using CellPose

d. Save Images and Masks in two separate folders

e. Use matlab code to create individual cell images, centered and noise free provided separately for datasets.




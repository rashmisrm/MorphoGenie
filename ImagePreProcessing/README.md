# Image Pre-processing 



## Image preprocessing pipeline involves several key steps to prepare images for analysis:

1. *Segmentation*:
    - For single-cell images captured using a high-speed imaging flow cytometer, intensity threshold-based segmentation is used (Lung Cancer, Cell Cycle       datasets). 
    - In contrast, for images of cells in culture plates with multiple cells in the field of view, Cellpose (Cellpose v2.1.1) is utilized for batch              processing of image segmentation, specifically leveraging the 'cyto2' model for segmenting images  (Cell Painting dataset and EMT). 

2. *Background noise removal*: Noise is removed while preserving cell body information.

3. *Center cropping and resizing*: Images are center-cropped and resized to 256 x 256 pixels.

4. *Cell alignment*: Cells are aligned to the center of the image frame to prevent positional features from influencing analysis.



a. _Download and extract the raw data files_: [Link](https://hkuhk-my.sharepoint.com/my?id=%2Fpersonal%2Frashmism%5Fhku%5Fhk%2FDocuments%2FMorphoGenieLink%2FRawDatasets&csf=1&web=1&CID=ee3d57ad%2D949c%2D4516%2D85d6%2D3257a6467d3b&FolderCTID=0x0120007AB2941E62AA1B49B2FF62BCAA405A51). 
b. _Copy relevant files_: Move the extracted raw data files into the [Link]() folder, ensuring the correct files are used with the pre-processing code.

The code for these steps is organized by dataset in individual folders. This approach allows for dataset-specific customization while maintaining a structured pipeline.
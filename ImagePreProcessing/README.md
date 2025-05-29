### Image Pre-processing 



#Image preprocessing pipeline involves several key steps to prepare images for analysis:

1. *Segmentation*:
    - For single-cell images captured using a high-speed imaging flow cytometer, intensity threshold-based segmentation is used (Lung Cancer, Cell Cycle       datasets). 
    - In contrast, for images of cells in culture plates with multiple cells in the field of view, Cellpose (Cellpose v2.1.1) is utilized for batch              processing of image segmentation, specifically leveraging the 'cyto2' model for segmenting images  (Cell Painting dataset and EMT). 

2. *Single-cell image generation*: Images are cropped into single-cell images.

3. *Background noise removal*: Noise is removed while preserving cell body information.

4. *Center cropping and resizing*: Images are center-cropped and resized to 256 x 256 pixels.

5. *Cell alignment*: Cells are aligned to the center of the image frame to prevent positional features from influencing analysis.



1. _Download the raw data files_: Get the files from the specified source 
2. _Extract the files_: If the files are compressed (e.g., zip, tar, or gzip), extract them to a folder.
3. _Copy relevant files_: Move the extracted raw data files into the 'ImagePreProcessing' folder, ensuring the correct files are used with the pre-processing code.

The code for these steps is organized by dataset in individual folders. This approach allows for dataset-specific customization while maintaining a structured pipeline.
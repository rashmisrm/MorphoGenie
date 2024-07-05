
# Downstream Analysis: Visualization, cell type/state classification  
Data is visualized in 2 dimensions using UMAP

## Lung Cancer Dataset
The Lung Cancer dataset obtained from a high throughput QPI System called Multi-ATOM. MorphoGenie could delineate three major histologically differentiated subtypes of lung cancer celllines H1975, H2170, H69.

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/LC.png)


### Cell Painting Assay Dataset
CPA dataset employed here is a subset of BBBC022, which is a publicly available fluorescence image dataset. The images consist of U2O2 cells treated with one of the 1600 bioactive compounds. In this dataset, images consisting of 5 channels tagged with 6 dyes characterizing 7 organells (nucleus, golgi-complex, mitochondria, nucleoli, cytoplasm, actin, endoplasmic reticulum) with 20x magnification.

![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/CPA.png)


For datasets the reveal continuous progressions, or develop into pathways, MorphoGenie uses VIA to perform trajectory inference.

## Cellular Progression Tracking
Through the integration of MorphoGenie and a trajectory inference technique called 
[StaVia](https://www.biorxiv.org/content/10.1101/2024.01.29.577871v1) forms a robust framework for providing holistic visual understanding of the continuous cellular processes with different complexities, as EMT and cell cycle progression studied in this work.


### Epithelial to Mesenchymal Transition (EMT)
EMT is a critical process underlying various biological phenomena, including embryonic development, tissue regeneration, and cancer progression. The disentangled representations derived from VIM-RFP-expressing cell images by MorphoGenie can be visualized using a trajectory inference tool called StaVia 
![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/Animate.gif)



### Cell Cycle 
Cell cycle dataset is imaged using another novel, QPI technique called Free-space Angular-Chirp-Enhanced Delay (FACED). MorphoGenie reveals new insights and heterogeneities in biological processes for cell cycle dataset of  MB231 celltypes.


![](https://github.com/rashmisrm/MorphoGenie/blob/main/Figures/AnimateCCy.gif)

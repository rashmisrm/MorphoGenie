# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:38:18 2022

@author: Kevin Tsia
"""


import random
import numpy as npana
import argparse
import os
from os import path
from tqdm import tqdm
import time
import copy
import numpy as np

import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints_test import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval_test import DisentEvaluator, Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler
)
import testDataset_results


from scipy.stats import zscore
Dir='Z:\COVID-FTP\Rashmi\ID-GANCheckpoints'
#Feature correlation and hierarchical clustering
#Feat=pd.read_csv('./outputs/TraversalLoop/TravAverageHeatmapTables/1-50Beta=1.csv')
#Feat=pd.read_csv('./outputs/TraversalLoop/TravAverageHeatmapTables/1-50Beta=8.csv')
#Feat=pd.read_csv(Dir+'/outputs/TraversalLoop/TravAverageHeatmapTables/1-50Beta=4.csv')

#Feat=pd.read_csv('./outputs/TraversalLoop/TravAverageHeatmapTables/1-25Gamma=6p3.csv')
#Feat=pd.read_csv(Dir+'./outputs/TraversalLoop/TravAverageHeatmapTables/1-32Gamma=15.csv')
#Feat=pd.read_csv(Dir+'./outputs/TraversalLoop/TravAverageHeatmapTables/CellCycle-Gamma=30.csv')
#Feat=pd.read_csv(Dir+'./outputs/TraversalLoop/TravAverageHeatmapTables/CellCycle-Gamma=50.csv')

#LC
#Feat=pd.read_csv(Dir+'./outputs/TraversalLoop/TravAverageHeatmapTables/LC-Gamma=25.csv')

#Vero
#Feat=pd.read_csv(Dir+'./outputs/TraversalLoop/TravAverageHeatmapTables/Vero-Gamma=30.csv')

#
Dir='C:/Users/Rashmi/WorkingLibs/idgan-master_Jan/outputs/TraversalLoop/CellCycle-50/'
Feat=pd.read_csv(Dir+'MeanVarHeatMap.csv')
Feat=Feat.loc[(Feat!=0).any(axis=1)]
sns.set(font_scale = 5)
#hierarchical Clustering
sns.set(font_scale = 10 )


def absolute_maximum_scale(series):
    return series / series.abs().max()



from scipy.stats import zscore
#C_10dim=pd.DataFrame(cdim_gan)
normed_matrix = zscore(Feat)
normed_matrix= pd.DataFrame(normed_matrix, columns=Feat.columns)

normalized_df=(Feat-Feat.mean())/Feat.std()

normalized_df=(Feat-Feat.min())/(Feat.max()-Feat.min())

#Remove columns with all zero values

normalized_df=normalized_df.dropna(1)



#corrmat=normed_matrix.corr(method ='pearson')

sns.set(font_scale = 20 )
plt.figure(figsize=(140,160))
CorrMat=normalized_df.corr(method ='pearson')




g=sns.clustermap(CorrMat, figsize=(120,120),xticklabels=Feat.columns,linewidths=8,tree_kws=dict(linewidths=10),cmap="vlag")
#g=sns.clustermap(normalized_df, figsize=(120,120),xticklabels=Feat.columns,linewidths=8,tree_kws=dict(linewidths=10))



## Variance gap computation

# Compute a secondary heatmap from the traversal heat-map
Attri=0
j=0
Metric=0
maxVar=0

Dataset=''
if Dataset=='CellCycle':
#Attributes for Cellcycle
    Attr1=['Area', 'Volume', 'DryMass', 'DryMassDensity', 'DMDVarMean', 'DMDRadialDistribution', 'QPFiberRadialDistribution', 
           'QPEntropyRadialDistribution']
    Attr2=['PhaseRange','PhaseMin', 'PhaseOrientationVar', 'DMDContrast3', 'DMDContrast4', 'QPEntropyCentroidDisplacement', 'DMDCentroidDisplacement', 
       'QPFiberCentroidDisplacement']
    Attr3=['QPEntropySkewness','DMDVarVariance','Circularity','Deformation', 'Orientation', 'AspectRatio', 'Eccentricity', 
           'DMDVarSkewness', 'PhaseArrangementSkewness']
    Attr4=['PeakPhase', 'PhaseSkewness', 'PhaseVar', 'PhaseCentroidDisplacement', 'PhaseOrientationKurtosis']
    Attr=[Attr1, Attr2, Attr3, Attr4]

elif Dataset=='LC':

#Attributes for LC
    Attr1=['Area', 'Volume', 'DryMass', 'DryMassDensity']
    Attr2=['DMDRadialDistribution', 'QPEntropyRadialDistribution']
    Attr3=['PhaseRange','PhaseMin', 'PhaseKurtosis', 'PhaseSkewness', 'PeakPhase', 'PhaseVar']
    Attr4=['MeanPhaseArrangement', 'PhaseArrangementSkewness','PhaseArrangementVar','PhaseOrientationVar', ]
    Attr5=['DMDVarSkewness','DMDVarKurtosis','DMDVarVariance']
    Attr6=['PhaseCentroidDisplacement','DMDCentroidDisplacement','PhaseCentroidDisplacement','QPEntropyCentroidDisplacement']
    Attr=[Attr1, Attr2, Attr3, Attr4, Attr5, Attr6]

elif Dataset=='Vero':

#Attributes for Vero
    Attr1=['Area', 'Volume', 'DryMass', 'DryMassDensity', 'DMDRadialDistribution', 'QPEntropyRadialDistribution']
    Attr2=['PhaseRange','PhaseMin', 'PhaseOrientationVar', 'PhaseKurtosis', 'PhaseSkewness', 'PeakPhase', 'PhaseVar']
    Attr3=['MeanPhaseArrangement', 'PhaseArrangementSkewness','PhaseArrangementVar']
    Attr4=['DMDVarSkewness','DMDVarKurtosis','DMDVarVariance']
    Attr5=[['PhaseCentroidDisplacement','DMDCentroidDisplacement','PhaseCentroidDisplacement','QPEntropyCentroidDisplacement']]
    
elif Dataset=='Other':
    Attr1=['Area', 'Volume', 'DryMass', 'DryMassDensity', 
           'DMDRadialDistribution']
    Attr2=['PhaseRange','PhaseMin', 'PhaseOrientationVar', 'PhaseVar', 'PeakPhase', 'PhaseSkewness', 
           'PhaseCentroidDisplacement', 'PhaseOrientationKurtosis']
    Attr3=['Circularity','Deformation', 'Orientation', 'AspectRatio', 'Eccentricity']
    Attr4=['DMDCentroidDisplacement', 'QPEntropyCentroidDisplacement', 'DMDVarMean','DMDVarVariance', 
           'QPEntropyMean', 'QPEntropySkewness', 'QPEntropyKurtosis', 'QPEntropyVar', 'QPFiberCentroidDisplacement', 
           'QPFiberPixelUpperPercentile' , 'DryMassVar', 'DryMassSkewness', 'DMDContrast1', 'DMDContrast2',
           'DMDContrast3', 'DMDContrast4']

    
else:
    Attr1=['PhaseOrientationVar', 'DMDContrast2','QPFiberPixelMedian','QPFiberRadialDistribution', 'DMDVarVariance',
           'Circularity', 'Deformation', 'AspectRatio', 'QPEntropyRadialDistribution', 'DMDRadialDistribution', 'Area', 'Volume',
           'QPEntropyMean', 'DMDVarSkewness', 'DMDVarKurtosis']
    Attr2=[ 'PhaseVar','QPEntropyVar','PhaseSkewness','QPEntropyCentroidDisplacement', 'MeanPhaseArrangement','PhaseArrangementVar',
           'PhaseArrangementSkewness']


    Attr3=['PhaseRange', 'PhaseMin', 'PhaseKurtosis', 'PhaseOrientationKurtosis', 'QPFiberPixelUpperPercentile']
    

    Attr=[Attr1, Attr2, Attr3]
'''

AttrValArray=[]
AttrDim=[]
for dimN in range (0, len(normalized_df)):
    AttrValArray=[]
    MeanAttN=0
    for Attri in Attr:
        for i in range(0, len(Attri)):
            AttrVal=normalized_df.at[dimN, Attri[i]]
            MeanAttN=MeanAttN + AttrVal
        MeanAttN=MeanAttN/len(Attri)

        AttrValArray.append(MeanAttN)
    AttrDim.append(AttrValArray)
AttrDim=np.vstack(AttrDim)    
a = sns.heatmap(AttrDim)

g=sns.clustermap(AttrDim, figsize=(120,120),linewidths=8,tree_kws=dict(linewidths=10),cmap="YlGnBu")            
maxVar1=0    
maxVar2=0
#Compute disentanglement metric from the secondary heatmap
for i in range(0, len(Attr)):
    for j in range(0, len(AttrDim)):
        MaxVar1=max(AttrDim[:,i])
        max_ind=AttrDim[:,i].tolist().index(MaxVar1)
        if (j != max_ind):
            MaxVar2=max(maxVar2, AttrDim[j,i])
    #Metric_k=AttrDim[:,j]
    Metric_k= MaxVar1-MaxVar2
    Metric=Metric+Metric_k
    
print('Metric=')
print(Metric/len(Attr)) 

'''

##### COLOR CODED CLusterMAP
g1=sns.clustermap(normalized_df, figsize=(140,140),xticklabels=Feat.columns,linewidths=8,tree_kws=dict(linewidths=10))

df_New=normalized_df
# First Remove non useful rows

for i, row in normalized_df.iterrows():
    var=(np.var(row) )
    mean=np.mean(row)
    print(var)
    if(var < 0.06):
        if(mean < 0.5):
            df_New=df_New.drop(i)

sns.set(font_scale = 20 )

g2=sns.clustermap(df_New, figsize=(140,140),xticklabels=Feat.columns,linewidths=8,tree_kws=dict(linewidths=15))


           
ClusteredVarMar=g2.data2d
df_Color=ClusteredVarMar
for i, row in df_New.iterrows():
    for j, col in row.iteritems():
        if (col > 0.9):
            print(col)
            if j in Attr1:
                df_Color.loc[i,j]=1
            elif j in Attr2:
                df_Color.loc[i,j]=2
    
            elif j in Attr3:
                df_Color.loc[i,j]=3
            '''
            elif j in Attr4:
                df_Color.loc[i,j]=4
            elif j in Attr5:
                df_Color.loc[i,j]=5
                '''

        else:
            nVal=0
            df_Color.loc[i,j]=nVal
            

#specify size of heatmap
fig, ax = plt.subplots(figsize=(180, 100))

#create heatmap
sns.heatmap(df_Color, linewidths=.3, cmap='magma')

A = [[83.85, 98.1, 94.76, 91.79, 78.3], 
    [80.02, 98.5, 96.46, 92.001, 77.4],
    [80.79, 94.01, 94.76, 92.03, 79.3],
    [80.18, 98.53, 96.32, 92.55, 78.4],
    [79.9, 98.43, 97.17, 95.28, 80.2]]

import numpy as np 
from pandas import *
import seaborn as sns
Index= ['CellCycle','LungCancer','LiveCell','EMT','CPA']
Cols = ['CellCycle','LungCancer','LiveCell','EMT','CPA']

df = DataFrame(A, index= Index, columns=Cols)
sns.heatmap(df, annot=True)
sns.heatmap(df, cmap='YlGnBu', linewidths=0.5, annot=True, fmt=".2")

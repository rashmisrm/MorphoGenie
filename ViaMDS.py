# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:10:42 2023

@author: Kevin Tsia
"""

import pyVIA.core as via
import numpy as np
import pandas as pd
from datetime import datetime
import core_working_rw2 as via
import matplotlib.pyplot as plt


def plot_phate(data, labels, n_pcs = 10, knn_init=2,title_addon = ''):
    import phate
    import os.path
    print(os.path.abspath(phate.__file__))
    str_date = str(str(datetime.now())[-3:])
    print(f'{datetime.now()}\tstart Phate')
    knn_init=knn_init
    phate_op = phate.PHATE(n_pca=None, n_jobs=-1, knn=knn_init)
    embedding_phate = phate_op.fit_transform(data)
    via.plot_scatter(embedding=embedding_phate, labels=labels,title='phate '+title_addon)
    plt.show()

def plot_umap(data, labels, title_addon = ''):
    import umap
    print(f'{datetime.now()}\tstart umap')
    print(data.shape)

    embedding_umap = umap.UMAP(random_state=42, n_neighbors=100, init='random', min_dist=0.3).fit_transform(
        data)
    print('embedding_umap', embedding_umap.shape)
    via.plot_scatter(embedding=embedding_umap, labels=labels, title='umap ' +title_addon)
    plt.show()

def read_cc(dataset_name = 'CCy'):
    #dataset_name 'CCy' or 'EMT'
    print('dataset is', dataset_name)
    foldername = 'C:/Users/Kevin Tsia/VIA/EMT-CellCycle/'
    labels = pd.read_csv(foldername + 'Mix_Label'+dataset_name+'.csv')
    labels = labels.drop(['Unnamed: 0'], axis=1)
    data = pd.read_csv(foldername + 'Mix_Latent'+dataset_name+'.csv')
    data = data.drop(['Unnamed: 0'], axis=1)
    print(set(labels['0'].tolist()))
    print(data.head(-5))
    return data.values, labels['0'].tolist()

def run_via_emt(data, labels):

    q_memory = 1 #IGNORE THIS PARAMETER IN YOUR RUNS RASHMI
    random_seed = 0
    jac_std_global =1
    knn = 100
    embedding_type = 'via-mds'#'via-umap'
    neighboring_terminal_states_threshold =3
    cluster_graph_pruning_std = 0.15
    time_series = False
    root = ['Epithelial'] #[1]
    print(f'q memory {q_memory} at random_seed {random_seed}')

    v0 = via.VIA(data, true_label=labels, jac_std_global=jac_std_global,
                 dist_std_local=1, knn=knn, q_memory=q_memory,
                 cluster_graph_pruning_std=cluster_graph_pruning_std,
                 neighboring_terminal_states_threshold=neighboring_terminal_states_threshold,
                 too_big_factor=0.3, resolution_parameter=1,
                 root_user=root, dataset='group', random_seed=random_seed,
                 is_coarse=True, preserve_disconnected=False, pseudotime_threshold_TS=40, x_lazy=0.99,
                 do_gaussian_kernel_edgeweights=True,
                 alpha_teleport=0.99, edgebundle_pruning=cluster_graph_pruning_std, edgebundle_pruning_twice=False,
                 velo_weight=None,
                 time_series=time_series, time_series_labels=None, knn_sequential=None,
                 knn_sequential_reverse=None, t_diff_step=None, RW2_mode=False,
                 working_dir_fp='C:/Users/Kevin Tsia/VIA/EMT-CellCycle/RW2', do_compute_embedding=True,
                 embedding_type=embedding_type)
    v0.run_VIA()
    via.draw_piechart_graph(via_object=v0)
    U = pd.read_csv('C:/Users/Kevin Tsia/VIA/EMT-CellCycle/PCs/viamds_singlediffusion_pcs10_k20_milestones3000_kprojectmilestones15t_stepNone_knnmds15_kseqmds2_kseqNone_nps10_tdiffNone_randseed0_diffusionop25_RsMds0_008.csv')
    U=U.drop(['Unnamed: 0'], axis=1)
    v0.embedding = U.values
    plt.show()
    decay = 0.7
    i_bw = 0.02
    global_visual_pruning = 0.5

    fig, ax = via.draw_sc_lineage_probability(via_object=v0, marker_lineages=[18,21,30])

    fig, ax = via.plot_edge_bundle(via_object=v0, lineage_pathway=[18,21,30],linewidth_bundle=1, alpha_bundle_factor=1, headwidth_bundle=0.2,
                    size_scatter=0.5, alpha_scatter=0.2 )
    fig.set_size_inches(25, 15)
    fig, ax = via.plot_edge_bundle(via_object=v0, n_milestones=50, decay=decay, initial_bandwidth=i_bw,
                                   linewidth_bundle=1.0, alpha_bundle_factor=1.5,
                                   cmap='plasma_r', facecolor='white', size_scatter=1, alpha_scatter=0.2,
                                   global_visual_pruning=global_visual_pruning,
                                   scale_scatter_size_pop=True, fontsize_labels=8,
                                   extra_title_text='EStage decay:' + str(decay) + ' bw:' + str(
                                       i_bw) + 'globalVisPruning:' + str(global_visual_pruning), headwidth_bundle=0.1,
                                   text_labels=False, sc_labels=labels)
    plt.show()


    plt.show()
    return

def main():
    dataset_name = 'EMT'
    #dataset_name='CCy'
    data, labels = read_cc(dataset_name=dataset_name)
    plot_umap(data,labels, title_addon = dataset_name)
    #plot_phate(data, labels, title_addon = dataset_name, knn_init=10)
    #run_via_emt(data=data,labels=labels)

if __name__ == '__main__':
    main()

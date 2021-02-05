"""
・学習済みモデルに対して、学習データ(既知クラスのみ)、テストデータ（既知クラス、未知クラス）を通したときの正規化されたｆ
をUMAPで２次元に描画
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import sys
import umap



#---UMAP用パラメータ----
n_neighbors = 100 #15
min_dist = 0.0
n_components = 2 #何次元に変換するか
is_supervised = False #正解ラベルを使用するならTrue #時間かかるので注意
random_seed = 42
low_memory = False #メモリ節約するか
#--------------------


def makeumapfitter(f, label):
    mapper = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        n_components = n_components,
        random_state=random_seed,
        transform_seed=random_seed,
        low_memory=low_memory
        )
    fitter = mapper.fit(f, y=label) if is_supervised else mapper.fit(f)
    f_embeded  = fitter.embedding_
    
    return fitter, f_embeded

def makeumap(numclass, fsize, train_f, train_label, test_ind_f, test_ind_label, test_ood_f, test_ood_label, savedir, test_ind_result, test_ood_result):
    
    #Train dataの一部(ランダム選択)でUMAPの次元削減器を作成
    if n_components != fsize:
        logging.info(f'UMAPing for train data...')
        fitter, train_f_embeded = makeumapfitter(train_f, train_label)
        logging.info(f'UMAPing done.')
    else:
        train_f_embeded = train_f

    #Testの既知データをUMAPで次元削減
    if n_components != fsize:
        logging.info(f'UMAPing for test(IND) data...')
        test_ind_f_embeded = fitter.transform(test_ind_f) #学習データを次元削減したfitterで次元削減
        logging.info(f'UMAPing done.')
    else:
        test_ind_f_embeded = test_ind_f

    #Testの未知データをUMAPで次元削減
    if n_components != fsize:
        logging.info(f'UMAPing for test(OOD) data...')
        test_ood_f_embeded = fitter.transform(test_ood_f) #学習データを次元削減したfitterで次元削減
        logging.info(f'UMAPing done.')
    else:
        test_ood_f_embeded = test_ood_f

    
    xmax = np.max([np.max(train_f_embeded[:,0]),np.max(test_ind_f_embeded[:,0]),np.max(test_ood_f_embeded[:,0])]) + 5
    xmin = np.min([np.min(train_f_embeded[:,0]),np.min(test_ind_f_embeded[:,0]),np.min(test_ood_f_embeded[:,0])]) - 5
    ymax = np.max([np.max(train_f_embeded[:,1]),np.max(test_ind_f_embeded[:,1]),np.max(test_ood_f_embeded[:,1])]) + 5
    ymin = np.min([np.min(train_f_embeded[:,1]),np.min(test_ind_f_embeded[:,1]),np.min(test_ood_f_embeded[:,1])]) - 5

    #指定可能な色一覧：https://pythondatascience.plavox.info/matplotlib/%E8%89%B2%E3%81%AE%E5%90%8D%E5%89%8D
    if numclass == 3:
        cmap = ListedColormap(['blue','green',  'red', 'black'])
        cmap_mild = ListedColormap(['lightblue','lightgreen',  'lightsalmon', 'black'])
    if numclass ==4:
        cmap = ListedColormap(['blue','green',  'red', 'magenta', 'black'])
        cmap_mild = ListedColormap(['lightblue','lightgreen',  'lightsalmon', 'violet', 'black'])


    #学習データと既知を1枚に統合
    fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
    plt.scatter(train_f_embeded[:,0], train_f_embeded[:,1], s=120.0, c=train_label, marker='.', cmap=cmap_mild, vmin=-0.5, vmax=numclass+1-0.5, alpha=0.01) #学習データは薄く、大きめに表示
    oodindex = np.asarray(test_ind_result[:,2], dtype=np.int)==1 #OODと判定された画像のインデックス
    if np.sum(oodindex) > 0:
        plt.scatter(test_ind_f_embeded[oodindex, 0], test_ind_f_embeded[oodindex, 1], s=5.0, marker='x',c=test_ind_label[oodindex], cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5,  alpha=1.0) 
    indindex = np.asarray(test_ind_result[:,2], dtype=np.int)==0 #INDと判定された画像のインデックス
    if np.sum(indindex) > 0:
        plt.scatter(test_ind_f_embeded[indindex, 0], test_ind_f_embeded[indindex, 1], s=5.0, marker='.',c=test_ind_label[indindex], cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5,  alpha=1.0) 
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    #plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(numclass+2)-0.5)
    cbar.set_ticks(np.arange(numclass+1))
    legend = ['class' + str(i) for i in range(numclass)]
    legend.append('Unknown')
    cbar.set_ticklabels(legend)
    plt.title('Train and Known Data')
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(savedir, "out_trainIND.png"), format='png', dpi=600)

    #学習データと未知を1枚に統合
    fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
    plt.scatter(train_f_embeded[:,0], train_f_embeded[:,1], s=120.0, c=train_label, marker='.', cmap=cmap_mild, vmin=-0.5, vmax=numclass+1-0.5, alpha=0.01) #学習データは薄く、大きめに表示
    oodindex = np.asarray(test_ood_result[:,2], dtype=np.int)==1 #OODと判定された画像のインデックス
    if np.sum(oodindex) > 0:
        plt.scatter(test_ood_f_embeded[oodindex, 0], test_ood_f_embeded[oodindex, 1], s=5.0, marker='x', c=np.ones(np.sum(oodindex), dtype=np.int)*numclass, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0) 
    indindex = np.asarray(test_ood_result[:,2], dtype=np.int)==0 #INDと判定された画像のインデックス
    if np.sum(indindex) > 0:
        plt.scatter(test_ood_f_embeded[indindex, 0], test_ood_f_embeded[indindex, 1], s=5.0, marker='.', c=np.ones(np.sum(indindex), dtype=np.int)*numclass, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0) 
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    #plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(numclass+2)-0.5)
    cbar.set_ticks(np.arange(numclass+1))
    legend = ['class' + str(i) for i in range(numclass)]
    legend.append('Unknown')
    cbar.set_ticklabels(legend)
    plt.title('Train and Unknown Data')
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(savedir, "out_trainOOD.png"), format='png', dpi=600)

    #全体のデータを１枚に統合
    fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
    plt.scatter(train_f_embeded[:,0], train_f_embeded[:,1], s=120.0, c=train_label, marker='.', cmap=cmap_mild, vmin=-0.5, vmax=numclass+1-0.5, alpha=0.01) #学習データは薄く、大きめに表示
    oodindex = np.asarray(test_ind_result[:,2], dtype=np.int)==1 #OODと判定された画像のインデックス
    if np.sum(oodindex) > 0:
        plt.scatter(test_ind_f_embeded[oodindex, 0], test_ind_f_embeded[oodindex, 1], s=5.0, marker='x',c=test_ind_label[oodindex], cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5,  alpha=1.0) 
    indindex = np.asarray(test_ind_result[:,2], dtype=np.int)==0 #INDと判定された画像のインデックス
    if np.sum(indindex) > 0:
        plt.scatter(test_ind_f_embeded[indindex, 0], test_ind_f_embeded[indindex, 1], s=5.0, marker='.',c=test_ind_label[indindex], cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5,  alpha=1.0) 
    oodindex = np.asarray(test_ood_result[:,2], dtype=np.int)==1 #OODと判定された画像のインデックス
    if np.sum(oodindex) > 0:
        plt.scatter(test_ood_f_embeded[oodindex, 0], test_ood_f_embeded[oodindex, 1], s=5.0, marker='x', c=np.ones(np.sum(oodindex), dtype=np.int)*numclass, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0) 
    indindex = np.asarray(test_ood_result[:,2], dtype=np.int)==0 #INDと判定された画像のインデックス
    if np.sum(indindex) > 0:
        plt.scatter(test_ood_f_embeded[indindex, 0], test_ood_f_embeded[indindex, 1], s=5.0, marker='.', c=np.ones(np.sum(indindex), dtype=np.int)*numclass, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0) 
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    #plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(numclass+2)-0.5)
    cbar.set_ticks(np.arange(numclass+1))
    legend = ['class' + str(i) for i in range(numclass)]
    legend.append('Unknown')
    cbar.set_ticklabels(legend)
    plt.title('All Data Embedded via UMAP')
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(savedir, "out_all.png"), format='png', dpi=600)

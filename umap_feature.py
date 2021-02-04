"""
・学習済みモデルに対して、学習データ(既知クラスのみ)、テストデータ（既知クラス、未知クラス）を通したときの正規化されたｆ
・各クラスの中心ベクトル（正規化済み）
をUMAPで２次元に描画
"""
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import math
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap

from utils.customDataset import oct
from network import resnet_face

#疵種指定
train_folders = 'oct_ind1' #folder2labelで既知クラス扱いにするフォルダ（train用）
test_ind_folders = 'oct_ind1' #既知クラス扱いにするフォルダ（テスト用）
test_ood_folders = 'oct_ood1' #未知クラス扱いにするフォルダ（テスト用）

#使用データ
#dir_train = '/home/keisoku/work/ood2/data/oct/minidata'
#dir_test = dir_train
dir_train = '/home/keisoku/work/ood2/data/oct/train'
dir_test = '/home/keisoku/work/ood2/data/oct/test'

#学習済みモデル
trainedmodel = "/home/keisoku/work/ood_face/src/adacos_numfeature2/CP_best.pth"
backbone = 'resnet18'
metric = 'adacos'
svalue = None #autoならNone,固定値ならその値を入れる
pretrain = True #pretrainありならTrueにする
fsize = 2 #512 #抽出する特徴量のサイズ
in_channels = 3 #入力チャンネル数
numclass = 3 #既知クラスの個数

#保存場所。モデルで判定したnpyもここに保存（２回以降の実行ではモデル判定をスキップできる）.モデルか使用データを変えた場合はnpyを消すこと。
savedir = os.path.join(os.path.dirname(trainedmodel), 'umap1')

#---UMAP用パラメータ----
n_neighbors = 200 #15
min_dist = 0.0
n_components = 2 #何次元に変換するか
is_supervised = True #正解ラベルを使用するならTrue #時間かかるので注意
random_seed = 42
low_memory = False #メモリ節約するか
prune_rate = 0.2 #UMAPを作るときに使用するtrain dataの割合（間引き率）

#グラフ
is_dispeach = True #学習データ、テストデータのUMAP画像を単体で出力するならTrue
#指定可能な色一覧：https://pythondatascience.plavox.info/matplotlib/%E8%89%B2%E3%81%AE%E5%90%8D%E5%89%8D
if numclass == 3:
    cmap = ListedColormap(['blue','green',  'red', 'black'])
    cmap_mild = ListedColormap(['lightblue','lightgreen',  'lightsalmon', 'black'])
if numclass ==4:
    cmap = ListedColormap(['blue','green',  'red', 'magenta', 'black'])
    cmap_mild = ListedColormap(['lightblue','lightgreen',  'lightsalmon', 'violet', 'black'])


def makeumap(f, label, w=None):
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
    w_embeded = fitter.transform(w) if w is not None else None
    
    if w is None:
        return fitter, f_embeded
    else:
        return fitter, f_embeded, w_embeded

def pred(device, net, fsize, loader, n_data, prune_rate=None):
    net.eval()
    f = np.zeros([n_data, fsize]) if prune_rate is None else np.zeros([math.floor(n_data * prune_rate), fsize]) #特徴量ｆ
    label = np.zeros(n_data, dtype=np.int)  if prune_rate is None else np.zeros(math.floor(n_data * prune_rate), dtype=np.int)  #正解ラベル
    #w = np.zeros([numclass, fsize]) #各クラスの中心ベクトル
    step = 0
    with torch.no_grad():
        with tqdm(total = n_data, desc = 'Prediction of data', unit='img', leave = False) as pbar:
            for i, batch in enumerate(loader):
                imgs = batch['image']
                imgs = imgs.to(device=device, dtype=torch.float32)
                label[i]  = batch['label'].cpu().numpy() #onehotでない。正解クラスの数字単品。
                feature = net(imgs)
                f[i,:] = feature.cpu().numpy()
                #TODO: metric_fcにfeatureを入力して中心ベクトル、中心ベクトルとのコサインなども取得
                step += imgs.shape[0]
                if prune_rate is not None and step + 1 > math.floor(n_data * prune_rate):
                    break
                pbar.update(imgs.shape[0])
    
    return f, label

def make_args():
    parser = argparse.ArgumentParser(description='Train the classification net on images and labels', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load', action = 'store_false')
    parser.add_argument('--net',  type = str, default=backbone, choices=['resnet18','resnet34','resnet50','resnet101','resnet152'], dest='backbone')
    parser.add_argument('--metric', type=str, default=metric, choices=['adacos', 'arcface', 'sphereface', 'cosface', 'okatani', 'softmax'], dest='metric')
    parser.add_argument('--num_features', default=fsize, type=int, dest='num_features')
    parser.add_argument('--svalue',  type=float, nargs='?', default=None, dest='svalue')
    parser.add_argument('--mvalue',  type=float, nargs='?', default=None, dest='mvalue')
    
    return parser.parse_args()

if __name__ == '__main__':
    os.makedirs(savedir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dataset_train = oct(in_ch = in_channels, out_ch = numclass, img_dir = dir_train, labelfoldername = train_folders, hflip= False, train = False, pretrained = pretrain)
    dataset_test_ind = oct(in_ch = in_channels, out_ch = numclass, img_dir = dir_test, labelfoldername = test_ind_folders, hflip= False, train = False, pretrained = pretrain)
    dataset_test_ood = oct(in_ch = in_channels, out_ch = numclass, img_dir = dir_test, labelfoldername = test_ood_folders, hflip= False, train = False, pretrained = pretrain)

    train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_ind_loader = DataLoader(dataset_test_ind, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    test_ood_loader = DataLoader(dataset_test_ood, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    n_train = len(dataset_train)
    n_test_ind = len(dataset_test_ind)
    n_test_ood = len(dataset_test_ood)

    args = make_args()
    net = resnet_face.ResNet_face(args)
    net.to(device=device)
    net.load_state_dict(torch.load(trainedmodel))

    #Train dataの一部(ランダム選択)でUMAPの次元削減器を作成
    train_npz = os.path.join(savedir, 'train.npz')
    if not os.path.isfile(train_npz) :
        train_f, train_label = pred(device, net, fsize, train_loader, n_train, prune_rate=prune_rate)
        np.savez(train_npz.replace('.npz',''), train_f=train_f, train_label=train_label)
        logging.info(f'{train_npz} is saved.')
    else:
        logging.info(f'Loading {train_npz}...')
        loaddata = np.load(train_npz)
        train_f = loaddata['train_f']
        train_label = loaddata['train_label']
    
    if n_components != fsize:
        logging.info(f'UMAPing for train data...')
        fitter, train_f_embeded = makeumap(train_f, train_label)
        logging.info(f'UMAPing done.')
    else:
        train_f_embeded = train_f

    #Testの既知データをUMAPで次元削減
    test_ind_npz = os.path.join(savedir, 'test_ind.npz')
    if not os.path.isfile(test_ind_npz) :
        test_ind_f, test_ind_label = pred(device, net, fsize, test_ind_loader, n_test_ind)
        np.savez(test_ind_npz.replace('.npz', ''), test_ind_f=test_ind_f, test_ind_label=test_ind_label)
        logging.info(f'{test_ind_npz} is saved.')
    else:
        logging.info(f'Loading {test_ind_npz}...')
        loaddata = np.load(test_ind_npz)
        test_ind_f = loaddata['test_ind_f']
        test_ind_label = loaddata['test_ind_label']

    if n_components != fsize:
        logging.info(f'UMAPing for test(IND) data...')
        test_ind_f_embeded = fitter.transform(test_ind_f) #学習データを次元削減したfitterで次元削減
        logging.info(f'UMAPing done.')
    else:
        test_ind_f_embeded = test_ind_f


    #Testの未知データをUMAPで次元削減
    test_ood_npz = os.path.join(savedir, 'test_ood.npz')
    if not os.path.isfile(test_ood_npz) :
        test_ood_f, _ = pred(device, net, fsize, test_ood_loader, n_test_ood)
        np.savez(test_ood_npz.replace('.npz', ''), test_ood_f=test_ood_f)
        logging.info(f'{test_ood_npz} is saved.')
    else:
        logging.info(f'Loading {test_ood_npz}...')
        loaddata = np.load(test_ood_npz)
        test_ood_f = loaddata['test_ood_f']

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

    #それぞれ独立の描写
    if is_dispeach:
        #train
        fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
        plt.scatter(train_f_embeded[:,0], train_f_embeded[:,1], s=0.3, c=train_label, cmap=cmap_mild, vmin=-0.5, vmax=numclass+1-0.5, alpha=0.01)
        #plt.scatter(w_embeded[:,0], w_embeded[:,1], s=30.0, c=np.arange(numclass), marker='*', cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        #plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(numclass+2)-0.5)
        cbar.set_ticks(np.arange(numclass+1))
        legend = ['class' + str(i) for i in range(numclass)]
        legend.append('Unknown')
        cbar.set_ticklabels(legend)
        plt.title('Train Data Embedded via UMAP')
        fig.savefig(os.path.join(savedir, "out_train.png"), format='png', dpi=600)
        #test IND
        fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
        plt.scatter(test_ind_f_embeded[:,0], test_ind_f_embeded[:,1], s=0.3, c=test_ind_label, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        #plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(numclass+2)-0.5)
        cbar.set_ticks(np.arange(numclass+1))
        legend = ['class' + str(i) for i in range(numclass)]
        legend.append('Unknown')
        cbar.set_ticklabels(legend)
        plt.title('Test [Known] Data Embedded via UMAP')
        fig.savefig(os.path.join(savedir, "out_test_ind.png"), format='png', dpi=600)
        #test OOD
        fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
        plt.scatter(test_ood_f_embeded[:,0], test_ood_f_embeded[:,1], s=3.0, c=np.ones(n_test_ood, dtype=np.int)*numclass, marker='x', cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        #plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(numclass+2)-0.5)
        cbar.set_ticks(np.arange(numclass+1))
        legend = ['class' + str(i) for i in range(numclass)]
        legend.append('Unknown')
        cbar.set_ticklabels(legend)
        plt.title('Test [UNknown] Data Embedded via UMAP')
        fig.savefig(os.path.join(savedir, "out_test_ood.png"), format='png', dpi=600)


    #学習データと既知を1枚に統合
    fig, ax = plt.subplots(1, figsize=(8.0, 6.0))
    plt.scatter(train_f_embeded[:,0], train_f_embeded[:,1], s=120.0, c=train_label, marker='.', cmap=cmap_mild, vmin=-0.5, vmax=numclass+1-0.5, alpha=0.01) #学習データは薄く、大きめに表示
    plt.scatter(test_ind_f_embeded[:,0], test_ind_f_embeded[:,1], s=0.3, c=test_ind_label, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
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
    plt.scatter(test_ood_f_embeded[:,0], test_ood_f_embeded[:,1], s=3.0, c=np.ones(n_test_ood, dtype=np.int)*numclass, marker=',', cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
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
    plt.scatter(test_ind_f_embeded[:,0], test_ind_f_embeded[:,1], s=0.3, c=test_ind_label, cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
    plt.scatter(test_ood_f_embeded[:,0], test_ood_f_embeded[:,1], s=3.0, c=np.ones(n_test_ood, dtype=np.int)*numclass, marker=',', cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
    #plt.scatter(w_embeded[:,0], w_embeded[:,1], s=80.0, c=np.arange(numclass), marker='*', cmap=cmap, vmin=-0.5, vmax=numclass+1-0.5, alpha=1.0)
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

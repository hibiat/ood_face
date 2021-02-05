"""
学習でできたモデルに対して、OODか否かを判定する
・あるテスト画像をnet（backbone）に入れてfeatureを出す
・学習データも同様に入れてfeatureを出して、テスト画像のfeatureからの距離順にソート
・一定距離以内にあるクラスがなければ未知、あれば多数決で既知クラスを決める
"""

import numpy as np
from numpy import linalg as LA
import math
import torch
from tqdm import tqdm

def pred(phase, device, net, num_features, loader, n_data, prune_rate=None):
    net.eval()
    f = np.zeros([n_data, num_features]) if prune_rate is None else np.zeros([math.floor(n_data * prune_rate), num_features]) #特徴量ｆ
    label = np.zeros(n_data, dtype=np.int)  if prune_rate is None else np.zeros(math.floor(n_data * prune_rate), dtype=np.int)  #正解ラベル
    fname = [] #ファイル名。list
    step = 0
    with torch.no_grad():
        with tqdm(total = n_data, desc = 'Prediction of ' + phase + ' data', unit='img', leave = False) as pbar:
            bsize= loader.batch_size
            for batch in loader:
                imgs = batch['image']
                imgs = imgs.to(device=device, dtype=torch.float32)
                label[step:step+bsize]  = batch['label'].cpu().numpy() #onehotでない。正解クラスの数字単品。
                filenames = batch['filename']

                feature = net(imgs)
                f[step:step+bsize,:] = feature.cpu().numpy()
                fname.extend(filenames)

                step += imgs.shape[0]
                if prune_rate is not None and step + 1 > math.floor(n_data * prune_rate):
                    break
                pbar.update(bsize)
    
    return f, label, fname

def classify(device, net, numclass, train_loader, test_ind_loader, test_ood_loader, n_train, n_test_ind, n_test_ood, prune_rate, num_features, thr_dist, thr_minsamplenum):
    net.eval()
    feature_train, true_label_train, filename_train = pred('Train',  device, net, num_features, train_loader, n_train, prune_rate)
    feature_test_ind, true_label_test_ind, filename_test_ind = pred('Test(IND)',  device, net, num_features, test_ind_loader, n_test_ind)
    feature_test_ood, true_label_test_ood, filename_test_ood = pred('Test(OOD)',  device, net, num_features, test_ood_loader, n_test_ood)
    
    #test(既知クラス)の判定
    pred_indood_test_ind = np.zeros(n_test_ind, dtype=np.long)  #既知か未知かの判定結果。未知なら1、既知なら0が入る
    pred_inind_test_ind = np.zeros(n_test_ind, dtype=np.long)  #既知判定なら、既知のどのクラスか。各クラスの数字が入る
    
    for i in range(n_test_ind):
        testsample = feature_test_ind[i,:]
        dist = np.sqrt(LA.norm(np.ones((feature_train.shape[0], 1)) * testsample - feature_train, axis=1))
        sorted_index = np.argsort(dist) 
        neighbor_index = sorted_index[0:thr_minsamplenum] #距離が短いthr_minsamplenum個
        neighbor_dist = dist[neighbor_index] 
        neighbor_label = true_label_train[neighbor_index]
        
        if np.sum(neighbor_dist < thr_dist) == thr_minsamplenum:
            pred_indood_test_ind[i] = 0 #既知と判定
            clses, counts = np.unique(neighbor_label, return_counts=True)
            pred_inind_test_ind[i] = clses[np.argmax(counts)] #最近傍の既知クラスうち、最も多いクラスと判定
        else:
            pred_indood_test_ind[i] = 1 #未知と判定
            pred_inind_test_ind[i] = -1 
    
    #test(未知クラス)の判定
    pred_indood_test_ood = np.zeros(n_test_ood, dtype=np.long)  #既知か未知かの判定結果。未知なら1、既知なら0が入る
    pred_inind_test_ood = np.zeros(n_test_ood, dtype=np.long)  #既知判定なら、既知のどのクラスか。各クラスの数字が入る
    
    for i in range(n_test_ood):
        testsample = feature_test_ood[i,:]
        dist = np.sqrt(LA.norm(np.ones((feature_train.shape[0], 1)) * testsample - feature_train, axis=1))
        sorted_index = np.argsort(dist) #距離が短いもの順
        neighbor_index = sorted_index[0]
        neighbor_dist = dist[neighbor_index]
        neighbor_label = true_label_train[neighbor_index]

        if np.sum(neighbor_dist < thr_dist) == thr_minsamplenum:
            pred_indood_test_ood[i] = 0 #既知と判定
            clses, counts = np.unique(neighbor_label, return_counts=True)
            pred_inind_test_ood[i] = clses[np.argmax(counts)] #最近傍の既知クラスうち、最も多いクラスと判定
        else:
            pred_indood_test_ood[i] = 1 #未知と判定
            pred_inind_test_ood[i] = -1 
    
    #判定精度
    #(1)既知vs未知判定
    ind_ok = np.sum(pred_indood_test_ind==0) #正解が既知の画像を既知と判定できた個数（ただし既知クラス内の判定が正しいかは問わない）[TNに相当]
    ind_ng = n_test_ind - ind_ok # 正解が既知の画像を誤って未知と判定した個数 [FPに相当]

    ood_ok = np.sum(pred_indood_test_ood==1) #正解が未知の画像を正しく未知と判定できた個数[TPに相当]
    ood_ng = n_test_ood - ood_ok #正解が未知の画像を誤って既知と判定した個数[FNに相当]

    tprate = ood_ok / n_test_ood
    fprate = ind_ng / n_test_ind
    if ood_ok +ind_ng != 0:
        precision = ood_ok / (ood_ok +ind_ng)
    else:
        precision = 0.0
    
    if tprate + precision != 0:   
        fvalue_indood = (2.0 * tprate *precision) / (tprate+ precision)
    else:
        fvalue_indood = 0.0

    accuracy_indood = (ind_ok + ood_ok)/(n_test_ind + n_test_ood)

    #(2)既知と判定された正解が既知のサンプルのうち、正しく判定できた割合
    accuracy_inind =  0.0 if np.sum(pred_inind_test_ind != -1)==0 else np.sum(pred_inind_test_ind ==true_label_test_ind)/ np.sum(pred_inind_test_ind != -1)

    test_ind_result = np.hstack([np.array(filename_test_ind, dtype=np.str).reshape(len(filename_test_ind),1), np.expand_dims(true_label_test_ind,1), np.expand_dims(pred_indood_test_ind,1), np.expand_dims(pred_inind_test_ind,1)])
    test_ood_result = np.hstack([np.array(filename_test_ood, dtype=np.str).reshape(len(filename_test_ood),1), np.expand_dims(true_label_test_ood,1), np.expand_dims(pred_indood_test_ood,1), np.expand_dims(pred_inind_test_ood,1)])
    
    return fvalue_indood, accuracy_indood, accuracy_inind, test_ind_result, test_ood_result, feature_train, true_label_train,feature_test_ind, true_label_test_ind, feature_test_ood, true_label_test_ood






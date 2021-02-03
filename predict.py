"""
ODD(未知)テストデータの推論
・シングルラベルの前提
・精度の指標は既知クラス分類のF値の平均（これが高ければ保存するモデルを更新）
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from tqdm import tqdm
import torch

def predict(net,
            metric_fc,
            param_s,
            pred_loader,
            device,
            n_test, 
            out_channels, 
            isood, 
            criterion
            ):
    with tqdm(total = n_test, desc = 'Prediction', unit='img', leave = False) as pbar:
        net.eval()
        metric_fc.eval()
        counter = 0
        avg_test_loss = 0.0
        true_label_all = np.zeros([n_test])
        pred_label_softmax_all = np.zeros([n_test, out_channels]) #判定結果のsoftmax値の全クラス分
        pred_theta_all = np.zeros([n_test, out_channels])
        pred_scale_all = np.zeros([n_test, 1])
        pred_label_argmax_all = np.zeros([n_test]) #最大確信度を取ったクラス
        
        filename_all = [] #list

        bsize= pred_loader.batch_size
        with torch.no_grad():
            for batch in pred_loader:
                imgs = batch['image']
                true_label = batch['label'] #onehotでない。正解クラスの数字単品。
                filename = batch['filename']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_label = true_label.to(device=device, dtype=torch.long)
                feature = net(imgs)               
                pred_label_logit, pred_theta, f, w = metric_fc(feature) #label=None #pred_label_logitはsがかけられていない値であることに注意
                #岡谷研方式のみsがかけられているので、次の掛け算ではparam_s=1として計算
                param_s = param_s.to(device) 
                pred_label_logit *= param_s

                if isood == False:
                    loss  = criterion(pred_label_logit, true_label)
                    avg_test_loss += loss.item()
                
                true_label_all[counter:counter+bsize] = true_label.cpu().numpy()
                
                pred_label_softmax = torch.nn.functional.softmax(pred_label_logit, dim=1).cpu().numpy() #元netではlogitの出力なので、softmaxをつける
                pred_label_softmax_all[counter:counter+bsize,:] = pred_label_softmax
                
                pred_theta_all[counter:counter+bsize,:] = pred_theta.cpu().numpy()
                pred_scale_all[counter:counter+bsize,:] = param_s.cpu().numpy()
                
                pred_label_argmax_all[counter:counter+bsize] = np.argmax(pred_label_softmax, axis=1)

                filename_all.extend(filename) #listに複数要素を連結
                counter += bsize
                pbar.update(bsize)

    if isood == False:
        avg_test_loss /= n_test / bsize
        #auc= roc_auc_score(true_label_all, pred_label_sigmoid_all, average='micro', max_fpr=1.0)
        accuracy = np.sum(pred_label_argmax_all == true_label_all)/ n_test
        fvalue = f1_score(true_label_all, pred_label_argmax_all, average='macro') #既知クラス内の分類精度
    else:
        avg_test_loss = None
        fvalue = None
        accuracy = None

    return avg_test_loss, fvalue, accuracy, filename_all, true_label_all, pred_label_softmax_all, pred_theta_all, pred_scale_all 
    #avg_test_loss:float, auc:float, filename:list, true_label_all:ndarray, pred_label_softmax_all:ndarray 
    #pred_theta_all:ndarray, pred_scale_all:ndarray
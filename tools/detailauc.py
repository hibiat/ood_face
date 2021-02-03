"""
CP_best_ind_results.csv、CP_best_ood_results.csvを基に
各閾値での未検、過検を既知、未知クラスそれぞれで算出。
※真陽性率 (True positive rate) = 感度 (Sensitivity) =  検出率 (Recall) = TP/(TP+FN)
※偽陽性率 (False positive rate) = FP/(TN+FP)
※適合率/精度 (Precision) = TP/(TP+FP)

http://www.baru-san.net/archives/141

"""

import glob
import os 
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn import metrics


#[1] 既知クラス内の指標
def eval_insideind(numclass, dfind, dfood):
    #0~1まで0.01刻みの閾値での値
    tprate_inside_ind = np.zeros((numclass, 101)) #各既知クラスのTrue Positive rate(=Recall)
    fprate_inside_ind = np.zeros((numclass, 101)) #False Positive rate
    precision_inside_ind = np.zeros((numclass, 101)) #Precision
    fvalue_inside_ind = np.zeros((numclass, 101)) #F値
    auc_inside_ind = np.zeros(numclass) #AUC

    for i in range(numclass):
        trueheader = 'true'
        predheader = 'pred' + str(i)
        dfclass = dfind[[trueheader, predheader]]
        df_tpfn = dfclass[dfclass[trueheader] == i] #クラスiが正解のもの(TP+FN)
        df_tnfp = dfclass[dfclass[trueheader] != i] #クラスiが正解でないもの(TN+FP)

        tpfn = len(df_tpfn)
        tnfp = len(df_tnfp)

        for j, thr in enumerate(np.arange(0, 1.01, 0.01)):
            tp = len(dfclass[(dfclass[trueheader] == i) & (dfclass[predheader] > thr)])
            fp = len(dfclass[(dfclass[trueheader] != i) & (dfclass[predheader] > thr)]) 
            tpfp = len(dfclass[dfclass[predheader]>thr]) #クラスiと判定されたもの(TP+FP)

            tprate_inside_ind[i, j] = 0.0 if tpfn== 0.0 else tp / tpfn 
            fprate_inside_ind[i, j] =  0.0 if tpfp== 0.0 else fp / tnfp 
            precision_inside_ind[i, j] = 0.0 if tpfp== 0.0 else tp / tpfp

            if tprate_inside_ind[i, j] + precision_inside_ind[i, j] != 0:
                fvalue_inside_ind[i, j] = (2.0 * tprate_inside_ind[i, j] * precision_inside_ind[i, j])/(tprate_inside_ind[i, j] + precision_inside_ind[i, j])
            else:
                fvalue_inside_ind[i, j] = 0.0

        auc_inside_ind[i] = metrics.auc(fprate_inside_ind[i,], tprate_inside_ind[i,:])

    return tprate_inside_ind, fprate_inside_ind, precision_inside_ind, fvalue_inside_ind, auc_inside_ind
    


#[2] 未知クラスと既知クラスとの間の指標。既知クラスの判定があっているかは別。

def analyze_indood(dfind_array, dfood_array):
    #0~1まで0.01刻みの閾値での値
    tprate_ood = np.zeros(101) #未知クラスをpositiveと見たときのTrue Positive rate 
    fprate_ood = np.zeros(101) #既知クラスをnegativeと見たときのFalse Negative rate 
    precision_ood = np.zeros(101) #Precision
    fvalue_ood = np.zeros(101) #F値

    n_ind = dfind_array.shape[0]
    n_ood = dfood_array.shape[0]
    numclass = dfind_array.shape[1]

    for j, thr in enumerate(np.arange(0,1.01, 0.01)):
        thr_m = np.ones(dfind_array.shape) * thr
        pred = np.sum(dfind_array > thr_m, 1)
        pred = pred > np.zeros(pred.shape)
        ind_ok = np.sum(pred) #正解が既知の画像を既知と判定できた個数（ただし既知クラス内の判定が正しいかは問わない）[TNに相当]
        ind_ng = n_ind - ind_ok # 正解が既知の画像を誤って未知と判定した個数 [FPに相当]

        thr_m = np.ones(dfood_array.shape) * thr
        pred = np.sum(dfood_array <= thr_m, 1)
        pred = pred == numclass * np.ones(n_ood)
        ood_ok = np.sum(pred) #正解が未知の画像を正しく未知と判定できた個数[TPに相当]
        ood_ng = n_ood - ood_ok #正解が未知の画像を誤って既知と判定した個数[FNに相当]

        tprate_ood[j] = ood_ok / n_ood
        fprate_ood[j] = ind_ng / n_ind
        if ood_ok +ind_ng != 0:
            precision_ood[j] = ood_ok / (ood_ok +ind_ng)
        else:
            precision_ood[j] = 0.0
        
        if tprate_ood[j] + precision_ood[j] != 0:   
            fvalue_ood[j] = (2.0 * tprate_ood[j] *precision_ood[j]) / (tprate_ood[j] + precision_ood[j])
        else:
            fvalue_ood[j] = 0.0

    auc_ood= metrics.auc(fprate_ood, tprate_ood)

    return tprate_ood, fprate_ood, precision_ood, fvalue_ood, auc_ood

def eval_indood(numclass, dfind, dfood):
    predheader = ['pred' + str(i) for i in range(numclass)]
    dfind = dfind[predheader]
    dfood = dfood[predheader]
    
    dfind_array = np.array(dfind.values)
    dfood_array = np.array(dfood.values)

    tprate_ood, fprate_ood, precision_ood, fvalue_ood, auc_ood = analyze_indood(dfind_array, dfood_array)

    return tprate_ood, fprate_ood, precision_ood, fvalue_ood, auc_ood, dfind_array, dfood_array


def calc(numclass, indcsv, oodcsv, savefilename):
    dfind = pd.read_csv(indcsv)
    dfood = pd.read_csv(oodcsv)

    #[1]既知クラス内の分類精度評価
    tprate_inside_ind, fprate_inside_ind, precision_inside_ind, fvalue_inside_ind, auc_inside_ind = eval_insideind(numclass, dfind, dfood)

    # [2]未知クラスと既知クラスとの間の指標。既知クラスの判定があっているかは別。
    # つまり全クラスの確信度が閾値以下なら未知、どれか１つでも確信度が閾値以上なら既知と判定した場合の指標
    tprate_ood, fprate_ood, precision_ood, fvalue_ood, auc_ood , dfind_array, dfood_array = eval_indood(numclass, dfind, dfood)

    
    #グラフ描画

    #----------------既知クラス内のROC曲線（横軸FP、縦軸TP）--------------------
    fig = plt.figure()
    lw = 2
    colors = cycle([(0, 0, 1), (0, 0.5, 0),(1, 0, 0),(0, 0.75, 0.75),(0.75, 0, 0.75),'#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',])
    for i, color in zip(range(numclass), colors):
        plt.plot(fprate_inside_ind[i,:], tprate_inside_ind[i,:], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.4f})'
                ''.format(i, auc_inside_ind[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8, ncol=3)
    plt.legend(ncol=4, fontsize=4)
    fig.tight_layout()
    #plt.show()
    fig.savefig(savefilename.replace('.png', '_roc_inknowncls.png'), dpi=300)
    plt.close(fig)

    #---------------未知クラスと既知クラスとの間のROC曲線（横軸FP、縦軸TP）-----------------------
    fig = plt.figure()
    lw = 2

    plt.plot(fprate_ood, tprate_ood, lw=lw,
                label='ROC curve of Known vs Unknown classification (area = {0:0.4f})'
                ''.format(auc_ood))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8, ncol=3)
    plt.legend(ncol=1, fontsize=4)
    fig.tight_layout()
    #plt.show()
    fig.savefig(savefilename.replace('.png', '_roc_indood.png'), dpi=300)
    plt.close(fig)

    #----------- 各閾値での{既知クラス内+未知/既知クラス間} のTrue positive, False positove----------------
    fig = plt.figure()
    lw = 2
    colors = cycle([(0, 0, 1), (0, 0.5, 0),(1, 0, 0),(0, 0.75, 0.75),(0.75, 0, 0.75),'#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',])
    x = np.arange(0, 1.01, 0.01)
    for i, color in zip(range(numclass), colors):   
        plt.plot(x, tprate_inside_ind[i,:], '--', lw=lw, color = color, label=f'TP of class{i}')
        plt.plot(x, fprate_inside_ind[i,:], '-', lw=lw, color = color, label=f'FP of class{i}')

    plt.plot(x, tprate_ood, 'k--', lw=lw, label = 'TP of Known vs Unknown')
    plt.plot(x, fprate_ood, 'k-', lw=lw, label = 'FP of Known vs Unknown')
    plt.legend(ncol=2, fontsize=4)
    plt.xlabel('Threshold')
    plt.ylabel('True Positive, False Positive')
    fig.tight_layout()
    fig.savefig(savefilename.replace('.png', '_roc_thr.png'), dpi=300)
    plt.close(fig)
    
    #----------------既知クラス内のPR曲線（横軸Recall、縦軸Precision）--------------------
    fig = plt.figure()
    lw = 2
    colors = cycle([(0, 0, 1), (0, 0.5, 0),(1, 0, 0),(0, 0.75, 0.75),(0.75, 0, 0.75),'#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',])
    for i, color in zip(range(numclass), colors):
        plt.plot(tprate_inside_ind[i,:], precision_inside_ind[i,:], color=color, lw=lw,
                label='class {0}'
                ''.format(i))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(ncol=4, fontsize=4)
    fig.tight_layout()
    #plt.show()
    fig.savefig(savefilename.replace('.png', '_pr_indood.png'), dpi=300)
    plt.close(fig)

    #---------------未知クラスと既知クラスとの間のPR曲線（横軸Recall、縦軸Precision）-----------------------
    fig = plt.figure()
    lw = 2
    plt.plot(tprate_ood, precision_ood, lw=lw,
                label='PR curve of Known vs Unknown classification')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(ncol=4, fontsize=4)
    fig.tight_layout()
    #plt.show()
    fig.savefig(savefilename.replace('.png', '_pr_indood.png'), dpi=300)
    plt.close(fig)

    #--------------- 各閾値での既知クラス内のRecall, Precision--------------------
    fig = plt.figure()
    lw = 2
    colors = cycle([(0, 0, 1), (0, 0.5, 0),(1, 0, 0),(0, 0.75, 0.75),(0.75, 0, 0.75),'#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',])
    x = np.arange(0, 1.01, 0.01)
    for i, color in zip(range(numclass), colors):   
        plt.plot(x, tprate_inside_ind[i,:], ':', lw=lw, color = color, label=f'Recall of class{i}')
        plt.plot(x, precision_inside_ind[i,:], '-', lw=lw, color = color, label=f'Precision of class{i}')

    plt.plot(x, tprate_ood, 'k:', lw=lw, label = 'Recall of Known vs Unknown')
    plt.plot(x, precision_ood, 'k-', lw=lw, label = 'Precision of Known vs Unknowns')
    plt.legend(ncol=2, fontsize=4)
    plt.xlabel('Threshold')
    plt.ylabel('Recall, Precision')
    fig.tight_layout()
    fig.savefig(savefilename.replace('.png', '_pr_thr.png'), dpi=300)
    plt.close(fig)

    #---------------各閾値での既知クラス内のF値-----------------------
    fig = plt.figure()
    lw = 2
    colors = cycle([(0, 0, 1), (0, 0.5, 0),(1, 0, 0),(0, 0.75, 0.75),(0.75, 0, 0.75),'#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',])
    x = np.arange(0, 1.01, 0.01)
    for i, color in zip(range(numclass), colors):   
        plt.plot(x, fvalue_inside_ind[i,:], '-', lw=lw, color = color, label=f'F-value of class{i}')
        
    plt.plot(x, fvalue_ood, 'k-', lw=lw, label = 'F-value of Known vs Unknown')
    plt.legend(ncol=2, fontsize=4)
    plt.xlabel('Threshold')
    plt.ylabel('F-value')
    fig.tight_layout()
    fig.savefig(savefilename.replace('.png', '_f_thr.png'), dpi=300)
    plt.close(fig)


    #-----------------正解が既知、未知クラスの画像に対して、確信度の最大値をヒストグラムで表示
    maxconfind = np.max(dfind_array, axis=1)
    maxconfood = np.max(dfood_array, axis=1)
    
    fig = plt.figure()
    plt.hist([maxconfind, maxconfood], bins=20, range=(0,1), rwidth=0.8, label=['Known class', 'Unknown class'])
    plt.xlabel('Max Confidence')
    plt.ylabel('Histgram of max confidence')
    plt.legend(ncol=2, fontsize=4)
    fig.tight_layout()
    fig.savefig(savefilename.replace('.png', '_hist_maxconf.png'), dpi=300)
    plt.close()

    return fvalue_inside_ind, auc_inside_ind, fvalue_ood, auc_ood


if __name__ == '__main__':
    #特定フォルダ内で実行する場合はこちら
    # indcsv = '/home/keisoku/work/ood/src/OODmetric_1.0sig_0.0625contloss_m1.0_weight0_train_except_nofindinding_resnet18_b64_l0.001_pretrained/CP_best_ind_results.csv'
    # oodcsv = '/home/keisoku/work/ood/src/OODmetric_1.0sig_0.0625contloss_m1.0_weight0_train_except_nofindinding_resnet18_b64_l0.001_pretrained/CP_best_ood_results.csv'
    # savefilename = '/home/keisoku/work/ood/src/OODmetric_1.0sig_0.0625contloss_m1.0_weight0_train_except_nofindinding_resnet18_b64_l0.001_pretrained/roc_detail.png'

    #複数フォルダまとめて実行する場合はこちら
    dir = '/home/keisoku/work/ood/out/new'

    numclass = 3
    try:
        calc(numclass, indcsv, oodcsv, savefilename)
    except NameError:
        if dir is not None:
            listall = glob.glob(dir+'/**')
            dirall = [d for d in listall if os.path.isdir(os.path.join(dir, d))]
            for dirs in dirall:
                indcsv = os.path.join(dirs, 'CP_best_ind_results.csv')
                oodcsv = os.path.join(dirs, 'CP_best_ood_results.csv')
                savefilename = os.path.join(dirs, 'roc_detail.png')

                if os.path.isfile(indcsv) and os.path.isfile(oodcsv):
                    print(dirs)
                    calc(numclass, indcsv, oodcsv, savefilename)
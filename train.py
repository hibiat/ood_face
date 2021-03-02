"""
Adacos, arcface, cosfaceなどの顔認証アーキテクチャを入れたネットワークでの学習
既知vs未知クラス内の分類精度がより高いモデルを作る。精度の指標はAUC（これが高ければ更新）
"""

import argparse
import csv
from itertools import cycle, chain
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import os
import random
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from network import resnet_face, metrics
from utils.customDataset import oct
from classify import classify
from tools import makeumap


#dir_train = '/home/keisoku/work/ood2/data/oct/minidata' #クラスごとに分かれたフォルダがある場所
#dir_test =  dir_train
dir_train = '/home/keisoku/work/ood2/data/oct/train'
dir_test = '/home/keisoku/work/ood2/data/oct/test'
train_folders = 'oct_ind1_1000' #folder2labelで既知クラス扱いにするフォルダ（train用）
test_ind_folders = 'oct_ind1' #既知クラス扱いにするフォルダ（テスト用）
test_ood_folders = 'oct_ood1' #未知クラス扱いにするフォルダ（テスト用）
numclass = 4 #既知クラスの個数
in_channels = 3 #入力チャンネル数

def train_net(net,
              metric_fc,
              device,
              in_channels=3,
              numclass=0,
              epochs=5,
              batch_size=1,
              lr=0.01,
              save_cp=False,
              pretrained=False
              ):
    #writer = SummaryWriter(comment=f'_{get_args().savedir}')
    writer = SummaryWriter(log_dir=f'{get_args().savedir}')
    
    dataset_train = oct(in_ch = in_channels, out_ch = numclass, img_dir = dir_train, labelfoldername = train_folders, hflip= True, train = True, pretrained = get_args().load)
    dataset_test_ind = oct(in_ch = in_channels, out_ch = numclass, img_dir = dir_test, labelfoldername = test_ind_folders, hflip= False, train = False, pretrained = get_args().load)
    dataset_test_ood = oct(in_ch = in_channels, out_ch = numclass, img_dir = dir_test, labelfoldername = test_ood_folders, hflip= False, train = False, pretrained = get_args().load)
    
    if get_args().imbarance_care == False:
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    else:
        logging.info(f'Class Imbarance is considered.')
        class_count = [i for i in dataset_train.data_distribusion.values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        class_weights = class_weights.tolist()
        class_weights_all = []
        class_weights_all.extend([i] * j for i, j in zip(class_weights, class_count))
        class_weights_all = list(chain.from_iterable(class_weights_all)) #2次元を1次元に
        random.seed(0)
        random.shuffle(class_weights_all)
        random.seed(0)
        random.shuffle(dataset_train.ids)
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, sampler=weighted_sampler, num_workers=8, pin_memory=True) #shuffle=Falseにする
    
    test_ind_loader = DataLoader(dataset_test_ind, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_ood_loader = DataLoader(dataset_test_ood, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    n_train = len(dataset_train)
    n_test_ind = len(dataset_test_ind)
    n_test_ood = len(dataset_test_ood)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': metric_fc.parameters()}], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_args().epochs, eta_min=get_args().lr_min)
    
    global_step = 0

    #cudnn.benchmark = True

    logging.info(f'''Starting training:
        Device:          {device.type}
        Net:             {get_args().backbone}
        Metric:          {get_args().metric}
        Pretrained:      {pretrained}
        Input ch:        {in_channels}
        Output classes:  {numclass}
        Training size:   {n_train}
        Test size(Ind):  {n_test_ind}
        Test size(OOD):  {n_test_ood}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Save All CP:     {save_cp}
    ''')

    best_test = 0.0 #既知vs未知クラスの分類精度が最良のもの
    thr_dist = 1.0 #距離閾値の初期値
    for epoch in range(epochs):
        #for param_group in optimizer.param_groups:
        #    logging.info('LR:{}'.format(param_group['lr']))
        for phase in ['train', 'test']:           
            if phase == 'train':
                net.train()
                metric_fc.train()
                epoch_loss = 0.0
                
                with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                    for batch in train_loader:
                        imgs = batch['image']
                        true_label = batch['label'] #one-hotではない。ラベルの数字単品。
                        
                        # fig = plt.figure()
                        # sp1 = fig.add_subplot(1,2,1)
                        # sp2 = fig.add_subplot(1,2,2)
                        # sp1.imshow(imgs[1,0,:,:]) #ch0画像の表示
                        # sp2.imshow(imgs[1,1,:,:]) #ch1画像の表示                       
                        # plt.show()
                                                                   
                        imgs = imgs.to(device=device, dtype=torch.float32)
                        true_label = true_label.to(device=device, dtype=torch.long)  
                        feature = net(imgs)  
                        pred_label, _, param_s, _, _= metric_fc(feature, true_label) 

                        batch_loss = criterion(pred_label, true_label) 
                        epoch_loss += batch_loss.item()
                        #writer.add_scalar('Loss_batch/train', batch_loss.item(), global_step)

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
     
                        pbar.set_postfix(**{'loss (batch)': batch_loss.item()})
                        pbar.update(imgs.shape[0])
                        global_step += 1

                    epoch_loss /= n_train / get_args().batchsize                  
                    writer.add_scalar('Loss_epoch/train', epoch_loss, epoch)

            if phase == 'test':
                confm_indood, fvalue_indood, accuracy_indood, accuracy_inind, test_ind_result, test_ood_result, feature_train, true_label_train,feature_test_ind, true_label_test_ind, feature_test_ood, true_label_test_ood = classify(
                        device, net, train_loader, test_ind_loader, test_ood_loader, n_train, n_test_ind, n_test_ood, 
                        prune_rate=args.prune_rate, num_features=args.num_features, thr_dist= thr_dist, thr_minsamplenum=args.thr_minsamplenum
                        )
                
                writer.add_scalar('0_Performance/F_INDOOD', fvalue_indood, epoch) 
                writer.add_scalar('0_Performance/Accuracy_INDOOD', accuracy_indood, epoch) 
                writer.add_scalar('0_Performance/Accuracy_inIND', accuracy_inind, epoch)
                
                #距離閾値の探索
                dist_ind2train = np.asarray(test_ind_result[:,4:4+args.thr_minsamplenum], dtype=np.float)
                dist_ood2train = np.asarray(test_ood_result[:,4:4+args.thr_minsamplenum], dtype=np.float)
                med_dist_ind2train = np.median(dist_ind2train)
                med_dist_ood2train = np.median(dist_ood2train)
                thr_dist  = np.mean([med_dist_ind2train, med_dist_ood2train])
                writer.add_scalar('1_distthr', thr_dist, epoch) 
                #確信度の最大値のヒストグラムを表示
                hist_value = np.ravel(dist_ind2train)
                counts, limits = np.histogram(hist_value, bins=20)
                sum_sq = hist_value.dot(hist_value)
                writer.add_histogram_raw(
                    tag='Dist2train/fromIND',
                    min=hist_value.min(),
                    max=hist_value.max(),
                    num=len(hist_value),
                    sum=hist_value.sum(),
                    sum_squares=sum_sq,
                    bucket_limits=limits[1:].tolist(),
                    bucket_counts=counts.tolist(),
                    global_step=epoch)

                hist_value = np.ravel(dist_ood2train)
                counts, limits = np.histogram(hist_value, bins=20)
                sum_sq = hist_value.dot(hist_value)
                writer.add_histogram_raw(
                    tag='Dist2train/fromOOD',
                    min=hist_value.min(),
                    max=hist_value.max(),
                    num=len(hist_value),
                    sum=hist_value.sum(),
                    sum_squares=sum_sq,
                    bucket_limits=limits[1:].tolist(),
                    bucket_counts=counts.tolist(),
                    global_step=epoch)
                
                logging.info('[Test] F_INDOOD: {:.3f}, Accuracy_INDOOD: {:.3f}, Accuracy_inIND:{:.3f}, Dist:(IND){:.3f},(OOD){:.3f},(Ave){:.3f}'.format(fvalue_indood, accuracy_indood, accuracy_inind,med_dist_ind2train, med_dist_ood2train, thr_dist))

                #umapのグラフ作成
                makeumap.makeumap(numclass, args.num_features, 
                                        feature_train, true_label_train, 
                                        feature_test_ind, true_label_test_ind, 
                                        feature_test_ood, true_label_test_ood, 
                                        get_args().savedir,
                                        test_ind_result, test_ood_result)

                if best_test < fvalue_indood:
                    logging.info(f'Best model OOdvsIND updated (epoch {epoch + 1})!')
                    best_test = fvalue_indood
                    #学習済みモデル保存
                    torch.save(net.state_dict(),
                           os.path.join(get_args().savedir, 'CP_best.pth'))
                           
                    #結果保存
                    header = ['filename', 'True label', 'Pred(OOD=1,IND=0)', 'Pred(inIND)']
                    savefilename = os.path.join(get_args().savedir,'CP_best_ind') + '_results.csv'
                    with open(savefilename, 'w') as f:
                        writer_csv = csv.writer(f)
                        writer_csv.writerow(header)
                        writer_csv.writerows(test_ind_result)
                    savefilename = os.path.join(get_args().savedir,'CP_best_ood') + '_results.csv'
                    with open(savefilename, 'w') as f:
                        writer_csv = csv.writer(f)
                        writer_csv.writerow(header)
                        writer_csv.writerows(test_ood_result)
                    savefilename = os.path.join(get_args().savedir,'CP_bestinfo.csv')  
                    with open(savefilename, mode='w', ) as f:
                        writer_csv = csv.writer(f)
                        header = ['epoch',  'Fvalue(INDvsOOD)', 'Accuracy(INDvsOOD)', 'Accuracy(inIND)']
                        writer_csv.writerow(header)
                        data = [epoch + 1, fvalue_indood, accuracy_indood, accuracy_inind]
                        writer_csv.writerow(data)
                        writer_csv.writerow(['Conf Matrix',  'Pred(OOD)', 'Pred(IND)'])
                        writer_csv.writerow(['True(OOD)', confm_indood[0,0], confm_indood[0,1]])
                        writer_csv.writerow(['True(IND)', confm_indood[1,0], confm_indood[1,1]])


            if save_cp:
                torch.save(net.state_dict(),
                           os.path.join(get_args().savedir, f'CP_epoch{epoch + 1}.pth'))
                logging.info(f'Checkpoint {epoch + 1} saved !')

            
    writer.export_scalars_to_json("./tensorboard_writer.json")
    writer.close()



def get_args():
    parser = argparse.ArgumentParser(description='Train the classification net on images and labels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('--lr_min', type=float, 
                        help='Minimun of Cosine Learning rate', dest='lr_min', default=1e-3)
    #parser.add_argument('-f', '--load', dest='load', type=str, default=False,
    #                    help='Load model from a .pth file')
    parser.add_argument('--load', action = 'store_true', help='Load pretrained model')
    parser.add_argument('-o','--savedir', metavar='S', type = str, default='out', help='checkpoints is saved here', dest='savedir')
    parser.add_argument('-n','--net', metavar='N', type = str, default='resnet18', choices=['resnet18','resnet34','resnet50','resnet101','resnet152'], dest='backbone')
    parser.add_argument('--metric', type=str, dest='metric', default='adacos', choices=['adacos', 'arcface', 'sphereface', 'cosface', 'okatani', 'softmax'])
    parser.add_argument('--num_features', default=512, type=int, help='dimention of embedded features', dest='num_features')
    parser.add_argument('--svalue',  type=float, nargs='?', default=None,
                        help='Fixed Parameter s. Applicable to ArcFace,SphereFace,CosFace', dest='svalue')
    parser.add_argument('--mvalue',  type=float, nargs='?', default=None,
                        help='Fixed Parameter m. Applicable to AdaCos,ArcFace,SphereFace,CosFace', dest='mvalue')
    parser.add_argument('--prune_rate',  type=float, nargs='?', default=1.0,
                        help='Rate of pruning training dataset for classification', dest='prune_rate')
    parser.add_argument('--thr_minsamplenum',  type=int, nargs='?', default=10,
                        help='Min samples to say IND in the neighbor.', dest='thr_minsamplenum')
    parser.add_argument('--imbarance_care', action = 'store_true', help='Considering class imbalance')
           
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    try:
        os.makedirs(args.savedir, exist_ok=True)
        logging.info('Created directory for saving')
    except OSError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = resnet_face.ResNet_face(args)
    net.to(device=device)

    if args.metric == 'adacos':
        metric_fc = metrics.AdaCos(
        num_features=args.num_features, num_classes=numclass) if args.mvalue==None else metrics.AdaCos(
        num_features=args.num_features, num_classes=numclass, m=args.mvalue) #論文ではm=0
    elif args.metric == 'arcface':
        metric_fc = metrics.ArcFace(
        num_features=args.num_features, num_classes=numclass) if args.mvalue==None and args.svalue==None else metrics.ArcFace(
        num_features=args.num_features, num_classes=numclass, s=args.svalue, m=args.mvalue)
    elif args.metric == 'sphereface':
        metric_fc = metrics.SphereFace(
        num_features=args.num_features, num_classes=numclass) if args.mvalue==None and args.svalue==None else metrics.SphereFace(
        num_features=args.num_features, num_classes=numclass, s=args.svalue, m=args.mvalue)
    elif args.metric == 'cosface':
        metric_fc = metrics.CosFace(
        num_features=args.num_features, num_classes=numclass) if args.mvalue==None and args.svalue==None else metrics.CosFace(
        num_features=args.num_features, num_classes=numclass, s=args.svalue, m=args.mvalue)
    elif args.metric == 'okatani':
        metric_fc = metrics.Okatani(
        num_features=args.num_features, num_classes=numclass)
    elif args.metric == 'softmax':
        metric_fc = metrics.SoftMaxLayer(
        num_features=args.num_features, num_classes=numclass)

    metric_fc.to(device=device)

    try:
        train_net(net=net,
                  metric_fc=metric_fc,
                  device=device, 
                  in_channels=in_channels,
                  numclass=numclass,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  save_cp=False,
                  pretrained=args.load
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

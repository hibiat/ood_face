"""
Adacos, arcface, cosfaceなどの顔認証アーキテクチャを入れたネットワークでの学習
既知vs未知クラス内の分類精度がより高いモデルを作る。精度の指標はAUC（これが高ければ更新）
"""

import argparse
import csv
from itertools import cycle
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
import os
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
from utils.savefile import saveconf_ood
from predict import predict
from tools import detailauc


dir_train = '/home/keisoku/work/ood2/data/oct/minidata' #クラスごとに分かれたフォルダがある場所
dir_test =  dir_train
#dir_train = '/home/keisoku/work/ood2/data/oct/train'
#dir_test = '/home/keisoku/work/ood2/data/oct/test'
train_folders = 'oct_ind1' #folder2labelで既知クラス扱いにするフォルダ（train用）
test_ind_folders = 'oct_ind1' #既知クラス扱いにするフォルダ（テスト用）
test_ood_folders = 'oct_ood1' #未知クラス扱いにするフォルダ（テスト用）
numclass = 3 #既知クラスの個数
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
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
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
    best_test2 = 0.0 #既知内精度が最良のもの
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
                #既知クラス判定
                avg_test_loss, fvalue, accuracy, filename_all_ind, true_label_all_ind, pred_label_softmax_all_ind, pred_theta_all_ind, pred_scale_all_ind = predict(
                                                                                                                        net,
                                                                                                                        metric_fc,
                                                                                                                        param_s,
                                                                                                                        test_ind_loader, 
                                                                                                                        device, 
                                                                                                                        n_test_ind, 
                                                                                                                        numclass, 
                                                                                                                        isood=False,
                                                                                                                        criterion=criterion                                                                                                                      
                                                                                                                        )
                
                writer.add_scalar('Loss_epoch/test', avg_test_loss, epoch) 
                writer.add_scalar('inIND/F_epoch', fvalue, epoch) 
                writer.add_scalar('inIND/Accuracy_epoch', accuracy, epoch) 
                writer.add_scalar('ParamS_mean/IND', np.mean(pred_scale_all_ind), epoch) 
                #未知クラス判定
                _, _, _, filename_all_ood, true_label_all_ood, pred_label_softmax_all_ood, pred_theta_all_ood, pred_scale_all_ood = predict(
                                                                                                                    net, 
                                                                                                                    metric_fc,
                                                                                                                    param_s,
                                                                                                                    test_ood_loader, 
                                                                                                                    device, 
                                                                                                                    n_test_ood, 
                                                                                                                    numclass, 
                                                                                                                    isood=True,
                                                                                                                    criterion=criterion
                                                                                                                    )
                writer.add_scalar('ParamS_mean/OOD', np.mean(pred_scale_all_ood), epoch) 
                #未知、既知クラス間の2クラス判定精度
                _, _, _, _, auc_oodind = detailauc.analyze_indood(pred_label_softmax_all_ind, pred_label_softmax_all_ood)
                writer.add_scalar('OODvsIND/AUC_epoch', auc_oodind, epoch) 

                logging.info('Test loss: {:.4f}, inIND_fvalue: {:.4f}, inIND_accuracy:{:.3f}, OODvsIND_AUC:{:.4f}'.format(avg_test_loss, fvalue, accuracy, auc_oodind))


                if best_test < auc_oodind: #未知クラスとの分類精度
                    best_test = auc_oodind
                    logging.info(f'Best model OOdvsIND updated (epoch {epoch + 1})!')

                    #既知クラスの結果保存
                    #ファイル名、正解、確信度, cos類似度、係数sをcsv保存
                    savefilename_ind = os.path.join(get_args().savedir,'CP_best_ind') + '_results.csv'                   
                    saveconf_ood(numclass, filename_all_ind, true_label_all_ind, pred_label_softmax_all_ind, pred_theta_all_ind, pred_scale_all_ind, savefilename_ind)
                    #確信度の最大値のヒストグラムを表示
                    hist_value = np.max(pred_label_softmax_all_ind, axis=1)
                    counts, limits = np.histogram(hist_value, bins=20, range=(-1,1))
                    sum_sq = hist_value.dot(hist_value)
                    writer.add_histogram_raw(
                        tag='OODMaxConf/test_IND',
                        min=hist_value.min(),
                        max=hist_value.max(),
                        num=len(hist_value),
                        sum=hist_value.sum(),
                        sum_squares=sum_sq,
                        bucket_limits=limits[1:].tolist(),
                        bucket_counts=counts.tolist(),
                        global_step=epoch)


                    #OODクラスの結果保存
                    #ファイル名、正解、確信度, cos類似度、係数sをcsv保存
                    savefilename_ood = os.path.join(get_args().savedir,'CP_best_ood') +'_results.csv'                 
                    saveconf_ood(numclass, filename_all_ood, true_label_all_ood, pred_label_softmax_all_ood, pred_theta_all_ood, pred_scale_all_ood, savefilename_ood)                   
                    #OODの確信度平均を表示
                    ave_ood_conf = np.mean(pred_label_softmax_all_ood)
                    writer.add_scalar('OODAveConf_epoch/test_ood', ave_ood_conf, epoch)                   
                    #確信度の最大値のヒストグラムを表示
                    hist_value = np.max(pred_label_softmax_all_ood, axis=1)
                    counts, limits = np.histogram(hist_value, bins=20, range=(-1,1))
                    sum_sq = hist_value.dot(hist_value)
                    writer.add_histogram_raw(
                        tag='OODMaxConf/test_OOD',
                        min=hist_value.min(),
                        max=hist_value.max(),
                        num=len(hist_value),
                        sum=hist_value.sum(),
                        sum_squares=sum_sq,
                        bucket_limits=limits[1:].tolist(),
                        bucket_counts=counts.tolist(),
                        global_step=epoch)
                    
                    #未知vs既知クラスの指標評価,既知クラス内の指標評価
                    #グラフ作成
                    _, auc_inside_ind, _, _ = detailauc.calc(numclass, savefilename_ind, savefilename_ood, os.path.join(get_args().savedir,'eval.png'))
                    #保存
                    savefilename = os.path.join(get_args().savedir,'saveinfo.csv')  
                    with open(savefilename, mode='w', ) as f:
                        writer_csv = csv.writer(f)
                        header = ['epoch',  'OODvsIND_AUC','inIND_accuracy']
                        header.extend(['inIND_AUC_cls' + str(i) for i in range(numclass)])
                        writer_csv.writerow(header)
                        data = [epoch+1, auc_oodind, accuracy]
                        data.extend(auc_inside_ind)
                        writer_csv.writerow(data)
                    
                    #学習済みモデル保存
                    torch.save(net.state_dict(),
                           os.path.join(get_args().savedir, 'CP_best.pth'))


                if best_test2 < accuracy: #既知クラス内の判定精度
                    best_test2 = accuracy
                    logging.info(f'Best model inIND updated (epoch {epoch + 1})!')

                    #既知クラスの結果保存
                    #ファイル名、正解、確信度, cos類似度、係数sをcsv保存
                    savefilename_ind = os.path.join(get_args().savedir,'CP_best_ind') + '_results2.csv'                   
                    saveconf_ood(numclass, filename_all_ind, true_label_all_ind, pred_label_softmax_all_ind, pred_theta_all_ind, pred_scale_all_ind, savefilename_ind)

                    #OODクラスの結果保存
                    #ファイル名、正解、確信度, cos類似度、係数sをcsv保存
                    savefilename_ood = os.path.join(get_args().savedir,'CP_best_ood') +'_results2.csv'                 
                    saveconf_ood(numclass, filename_all_ood, true_label_all_ood, pred_label_softmax_all_ood, pred_theta_all_ood, pred_scale_all_ood, savefilename_ood)                   
                
                    #未知vs既知クラスの指標評価,既知クラス内の指標評価
                    #グラフ作成
                    _, auc_inside_ind, _, _  = detailauc.calc(numclass, savefilename_ind, savefilename_ood, os.path.join(get_args().savedir,'eval2.png'))
                    #保存
                    savefilename = os.path.join(get_args().savedir,'saveinfo2.csv')  
                    with open(savefilename, mode='w', ) as f:
                        writer_csv = csv.writer(f)
                        header = ['epoch',  'OODvsIND_AUC','inIND_accuracy']
                        header.extend(['inIND_AUC_cls' + str(i) for i in range(numclass)])
                        writer_csv.writerow(header)
                        data = [epoch+1, auc_oodind, accuracy]
                        data.extend(auc_inside_ind)
                        writer_csv.writerow(data)
                        
              
                    #学習済みモデル保存
                    torch.save(net.state_dict(),
                           os.path.join(get_args().savedir, 'CP_best2.pth'))
                    
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
    parser.add_argument('-s', '--svalue', metavar='PS', type=float, nargs='?', default=None,
                        help='Fixed Parameter s', dest='svalue') #指定しなけれbばｓは学習で自動探索、指定すればｓは固定
    

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
        num_features=args.num_features, num_classes=numclass)
    elif args.metric == 'arcface':
        metric_fc = metrics.ArcFace(
        num_features=args.num_features, num_classes=numclass)
    elif args.metric == 'sphereface':
        metric_fc = metrics.SphereFace(
        num_features=args.num_features, num_classes=numclass)
    elif args.metric == 'cosface':
        metric_fc = metrics.CosFace(
        num_features=args.num_features, num_classes=numclass)
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

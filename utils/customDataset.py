"""
OCTを対象としたClassification用のデータセットクラス。
画像サイズの変換は以下の通り。
[高さ]
　(train、test共通) 
496pixにランダムクロップ後に248pixへリサイズする。496pix未満なら周囲をエッジの画素値で埋めて496pixにしてから248pixにリサイズする。
[幅]
　(train) 512pixにランダムクロップ後に256pixへリサイズする。もし512pix未満なら周囲をエッジの画素値で埋めて512pixにしてからリサイズする。
  (test)  512pixにセンタークロップ後に256pixへリサイズする。もし512pix未満なら周囲をエッジの画素値で埋めて512pixにしてからリサイズする。

"""

import csv
import glob
import itertools
import logging
import math
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from . import folder2label


class oct(Dataset):
    def __init__(self, in_ch, out_ch, img_dir, labelfoldername, hflip= False, train = False, pretrained = True):
        self.in_ch = in_ch
        self.out_ch = out_ch  
        self.img_dir = img_dir
        self.labelfoldername = labelfoldername
        self.hflip = hflip
        self.train = train
        self.pretrained = pretrained

        targetfolder = folder2label.get_allfoldername(self.labelfoldername) #画像を取得する対象フォルダを限定
        targetfile = []
        self.data_distribusion = {}
        for folder in targetfolder:
            files = glob.glob(os.path.join(self.img_dir, folder) + '/*.jpeg')
            targetfile.append(files)
            self.data_distribusion[folder] = len(files)
            logging.info(f'"{folder}" has {len(files)} images.')
        self.ids = list(itertools.chain.from_iterable(targetfile)) #2次元リストを1次元に平坦化
    
        logging.info('Total %d images in %s.', len(self.ids), self.img_dir)

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):  
        ch0_filename = self.ids[index]
        self.ch0 = Image.open(ch0_filename).convert('RGB')
        w, h = self.ch0.size #PILではsizeの順番がw,hの順なので注意
        cropsize = (496, 512)
        resizesize = (248, 256)

        fill_top = math.ceil((cropsize[0]-h)/2) if cropsize[0]-h > 0 else 0
        fill_bottom = math.floor((cropsize[0]-h)/2) if cropsize[0]-h > 0 else 0
        fill_left = math.ceil((cropsize[1]-w)/2) if cropsize[1]-w > 0 else 0
        fill_right = math.floor((cropsize[1]-w)/2) if cropsize[1]-w > 0 else 0

        if self.train:
            proclist = [
                #transforms.Pad((fill_left, fill_top, fill_right, fill_bottom), padding_mode='edge'),#padding
                #transforms.RandomCrop(cropsize), #random crop
                transforms.Resize(resizesize), #resize into half
                transforms.ToTensor()
            ]
            if self.hflip:
                proclist.append(transforms.RandomHorizontalFlip()) #hflip
            trans = transforms.Compose(proclist)

        if not self.train:
            trans = transforms.Compose([
                #transforms.Pad((fill_left, fill_top, fill_right, fill_bottom), padding_mode='edge'),#padding
                #transforms.CenterCrop(cropsize), #center crop
                transforms.Resize(resizesize), #resize into half
                transforms.ToTensor()
            ]) 
        
        img_tensor = trans(self.ch0)
        
        assert img_tensor.shape[0] ==3, f"input image expects 3ch, but {ch0_filename} is {img_tensor.shape[0]} ch image."
        assert self.in_ch == 3, f"in_ch expects 3 but {self.in_ch}."

        if self.pretrained:
            img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor) #imagenetでの正規化
    
        dirname = os.path.basename(os.path.dirname(ch0_filename))
        label = folder2label.folder2label(self.labelfoldername, dirname)
        assert label is not None, f"label for {ch0_filename} is not found."
        label = torch.tensor(label, dtype = torch.long) #one-hotでなくラベルの数字単品であることに注意!!

        return {'image':img_tensor, 'label':label, 'filename':ch0_filename}



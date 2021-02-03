"""
dataset内の画像のサイズの分布を確認
"""
import collections
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

indir = "/home/keisoku/work/ood2/data/oct/ダウンロード元ファイル/version3/ZhangLabData/CellData/OCT/train/DRUSEN"

h = []
w = []
hmax = 0
hmin = 2048
wmax = 0
wmin = 2048

for imgfile in glob.glob(indir+'/**/*.jpeg', recursive=True):
    img = cv2.imread(imgfile)
    height, width, channel = img.shape
    h.append(height)
    w.append(width)
    if hmax < height: hmax = height  
    if hmin > height: hmin = height 
    if wmax < width: wmax = width 
    if wmin > width: wmin = width 

print(f'height:{hmin}~{hmax}, width:{wmin}~{wmax}')
print('Height')
print(collections.Counter(h))
print('width')
print(collections.Counter(w))


h_array = np.array(h)
w_array = np.array(w)

fig = plt.figure()
hf = fig.add_subplot(1,2,1)
hf.hist(h_array, bins=100, range=(200, 600))
hf.set_title('Histgram of Hight')
hw = fig.add_subplot(1,2,2)
hw.hist(w_array, bins=100, range=(600, 2000))
hw.set_title('Histgram of Width')
plt.show()
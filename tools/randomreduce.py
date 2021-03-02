"""
フォルダ内の画像を指定枚数にランダムに選択する
選択されなかった画像は別フォルダ(unused)に移動
"""

import glob
import os
import random
import shutil

dir = "/home/keisoku/work/ood2/data/oct/train/NORMAL100"
selectimgnum = 100 #選択する枚数

files = glob.glob(os.path.join(dir, '*.jpeg'))
random.shuffle(files)
savedir = os.path.join(dir, 'unused')
os.makedirs(savedir, exist_ok=True)

for i in range(selectimgnum, len(files)):
    shutil.move(files[i], savedir)








'''
CP_best_ind_results.csvから指定クラスの最大確信度のヒストグラムを描画
CP_best_ood_results.csvの最大確信度のヒストグラムも重ねて表示
'''

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

ind_results_csv = '/home/keisoku/work/ood2/src/out/ood_sauto_cont/CP_best_ind_results.csv'
ood_results_csv = '/home/keisoku/work/ood2/src/out/ood_sauto_cont/CP_best_ood_results.csv'
savefilename = '/home/keisoku/work/ood2/src/out/ood_sauto_cont/eval_confhist_cls2.png'
selectclass = 2 #ヒストグラムに使用するクラスNo. 'all'とすれば全クラス
numclass = 3

assert selectclass < numclass, f'selectclass({selectclass}) should be smaller than numclass({numclass})'
dfind = pd.read_csv(ind_results_csv)
dfood = pd.read_csv(ood_results_csv)

if selectclass == 'all':
    predheader = ['pred' + str(i) for i in range(numclass)]
else:
    predheader = ['pred' + str(selectclass)]

dfind= dfind[predheader]
dfood = dfood[predheader]
dfind_array = np.array(dfind.values)
dfood_array = np.array(dfood.values)

maxconfind = np.max(dfind_array, axis=1)
maxconfood = np.max(dfood_array, axis=1)

fig = plt.figure()
plt.hist([maxconfind, maxconfood], bins=20, range=(0,1), rwidth=0.8, label=['Known class(' + str(selectclass) +')', 'Unknown class'])
plt.xlabel('Max Confidence')
plt.ylabel('Histgram of max confidence')
plt.legend(ncol=2, fontsize=4)
fig.tight_layout()
fig.savefig(savefilename, dpi=300)
plt.close()










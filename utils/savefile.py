import csv
import numpy as np



#Normal用　結果全保存
def saveconf_normal(numclass, filename, true_label, pred_label, savefilename):
    with open(savefilename, 'w') as f:
        writer = csv.writer(f)
        header = ['filename', 'true']
        header.extend(list('pred'+ str(i) for i in range(numclass)))
        writer.writerow(header)
        savedata = np.hstack((np.expand_dims(true_label,1), pred_label))
        savedata= np.hstack((np.array(filename,dtype=np.str).reshape(len(filename),1), savedata)).tolist()
        writer.writerows(savedata)

#OOD用　結果全保存
def saveconf_ood(numclass, filename, true_label, pred_label, pred_theta, pred_scale, savefilename):
    with open(savefilename, 'w') as f:
        writer = csv.writer(f)
        header = ['filename', 'true']
        header.extend(list('pred'+ str(i) for i in range(numclass)))
        header.extend(list('pred_theta'+ str(i) for i in range(numclass)))
        header.extend(['pred_scale'])
        
        writer.writerow(header)
        savedata = np.hstack((np.expand_dims(true_label,1), pred_label, pred_theta, pred_scale))
        savedata= np.hstack((np.array(filename,dtype=np.str).reshape(len(filename),1), savedata)).tolist()
        writer.writerows(savedata)
    

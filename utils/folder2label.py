def switch_mapping(defecttype):
    if defecttype == 'oct_ind1':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2          #Diabetic macular edema;糖尿病黄斑浮腫
        }
        numclass = 3
    
    if defecttype == 'oct_ood1':
        mapping = {
        'DRUSEN': 3 #ドルーゼン(網膜の細胞から出る老廃物)
        }
        numclass = 1

    if defecttype == 'oct_temp1':
        mapping = {
        'DRUSEN_train': 0 
        }
        numclass = 1
    if defecttype == 'oct_temp2':
        mapping = {
        'DRUSEN_test': 0
        }
        numclass = 1

    if defecttype == 'oct_ind1few10':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN10': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した10枚
        }
        numclass = 4

    if defecttype == 'oct_ind1few100':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN100': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4
    
    if defecttype == 'oct_ind1few1000':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN1000': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind1few2000':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN2000': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind1few4000':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN4000': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4
    
    if defecttype == 'oct_ind1few8616':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind1_1000':
        mapping = {
        'NORMAL1000': 0,      #異常なし
        'CNV1000': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME1000': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN1000': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4
    
    if defecttype == 'oct_ind1_1000_few100':
        mapping = {
        'NORMAL1000': 0,      #異常なし
        'CNV1000': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME1000': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN100': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind1_1000_few10':
        mapping = {
        'NORMAL1000': 0,      #異常なし
        'CNV1000': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME1000': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN10': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind1_100':
        mapping = {
        'NORMAL100': 0,      #異常なし
        'CNV100': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME100': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN100': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4
        
    if defecttype == 'oct_ind1_100_cls3':
        mapping = {
        'NORMAL100': 0,      #異常なし
        'CNV100': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME100': 2       #Diabetic macular edema;糖尿病黄斑浮腫
        }
        numclass = 3

    if defecttype == 'oct_ind1_100_few10':
        mapping = {
        'NORMAL100': 0,      #異常なし
        'CNV100': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME100': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN10': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind1_100_few10to100':
        mapping = {
        'NORMAL100': 0,      #異常なし
        'CNV100': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DME100': 2,         #Diabetic macular edema;糖尿病黄斑浮腫
        'DRUSEN100_dupfrom10': 3 #ドルーゼン(網膜の細胞から出る老廃物) ランダム選択した100枚
        }
        numclass = 4

    if defecttype == 'oct_ind2':
        mapping = {
        'NORMAL': 0,      #異常なし
        'CNV': 1,         #choroidal neovascularization; 脈絡膜新生血管
        'DRUSEN': 2 #ドルーゼン(網膜の細胞から出る老廃物)
        }
        numclass = 3
    
    if defecttype == 'oct_ood2':
        mapping = {
        'DME': 3          #Diabetic macular edema;糖尿病黄斑浮腫
        
        }
        numclass = 1

    if defecttype == 'new_dataset':
        mapping = {
        'leison0': 0,   
        'leison1': 1,     
        'leison2': 2 
        }
        numclass = 3

    return mapping, numclass



def folder2label(defecttype, foldername):
    mapping, _ = switch_mapping(defecttype)
    label =mapping.get(foldername)
    assert label != None, 'Folder {0} Not Found'.format(foldername)
    return label

def get_numclass(defecttype):
    _, numclass = switch_mapping(defecttype)
    assert numclass != None, 'Classnum of defecttype {0} Not Found'.format(defecttype)
    return numclass

def get_allfoldername(defecttype):
    mapping, _ = switch_mapping(defecttype)
    return mapping.keys()

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

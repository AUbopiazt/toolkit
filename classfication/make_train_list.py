import os
import glob
import random
rootPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/person/'
txtroot = rootPath + 'train.lst'
f = open(txtroot, 'w')
path_vector = []
for root, dirs, files in os.walk(rootPath):
    for dir in dirs:
        imgpath = os.path.join(root, dir) + '/*'
        for imgname in glob.glob(imgpath):
            imgNameList = imgname.split('/')[-2:]
            suxf = dir.split('_')[0]
            # assert suxf in ['noropes', 'ropes', 'other'], suxf
            # if suxf == 'noropes':
            #     label = '1'
            # if suxf == 'ropes':
            #     label = '2'
            # if suxf == 'other':
            #     label = '0'
            # if suxf not in ['valother', 'valperson']:
            #     continue
            assert suxf in ['other', 'person'], suxf
            if suxf == 'person':
                label = '1'
            if suxf == 'other':
                label = '0'
            #path_label = imgNameList[0] + '/' + imgNameList[1] + ' ' + label + '\n'
            path_label = imgNameList[0] + '/' + imgNameList[1] + ' ' + label
            path_vector.append(path_label)
            #f.write(path_label)
random.shuffle(path_vector)
for text in path_vector:
    newtext = text + '\n'
    f.write(newtext)
f.close()


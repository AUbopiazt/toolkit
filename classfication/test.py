import os
import glob
import shutil

rootPath = '/media/aubopiazt/0BCB18210BCB1821/dataformate/p001/rope/'
saveroot = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/ropes'
for roots, dirs, files in os.walk(rootPath):
    for dir in dirs:
        imgPath = os.path.join(roots, dir)
        for subRoot, subDirs, subFiles in os.walk(imgPath):
            for subDir in subDirs:
                subImagePath = os.path.join(subRoot, subDir)
                for img in glob.glob(subImagePath + '/*.jpg'):
                    shutil.copy(img, os.path.join(saveroot, subDir))

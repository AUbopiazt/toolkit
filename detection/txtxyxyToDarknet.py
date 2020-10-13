import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import os
import shutil

excavator = ['sea_person', 'earth_person', 'person']
car = ['car']
truck = ['truck']
other = ['other']
wheel = ['wheel']
labelAll = excavator
rigthLabel = excavator
xmlPath = '/media/aubopiazt/reid/zt/objectDet/personOne/*.txt'
savepath = '/media/aubopiazt/reid/zt/objectDet/txtsave/'
if os.path.exists(savepath):
    shutil.rmtree(savepath)
os.mkdir(savepath)
#ss = len(glob.glob(jsonPath))
count1 = 0
count2 = 0
count3 = 0
for xmlfile in glob.glob(xmlPath):
    savetxtfile = xmlfile.replace('personOne', 'txtsave')
    imgfile = xmlfile.replace('txt', 'jpg')
    print(imgfile)
    txt = open(savetxtfile, 'w')
    img = Image.open(imgfile)
    W, H = img.size
    boxes = []

    with open(xmlfile, 'r') as f:
        txtread = f.readlines()
        for lines in txtread:
            label_id = -1
            txtText = lines.rstrip().split(' ')
            label = txtText[1]
            if label in rigthLabel:
                label_id = 0
            else:
                continue
            xmin = float(txtText[2])
            ymin = float(txtText[3])
            xmax = float(txtText[4])
            ymax = float(txtText[5])
            norm_w = abs(xmax - xmin) / W
            norm_h = abs(ymax - ymin) / H
            centerx = (xmin + xmax) / 2
            centery = (ymin + ymax) / 2
            centerx = centerx / W
            centery = centery / H
            txt.write(str(label_id) + ' ' + str(centerx) + ' ' + str(centery) + ' ' + str(norm_w) + ' ' + str(norm_h) + '\n')
    txt.close()
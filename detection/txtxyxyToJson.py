import json
#import pandas as pd
import os
import numpy as np
import glob
import shutil
from PIL import Image

originfile = '/media/aubopiazt/reid/zt/objectDet/personOne/*.txt'
saveJoson = '/media/aubopiazt/reid/zt/objectDet/savejson/'
if os.path.exists(saveJoson):
    shutil.rmtree(saveJoson)
os.mkdir(saveJoson)


excavator = ['sea_person', 'earth_person', 'person']
car = ['car']
truck = ['truck']
other = ['other']
wheel = ['wheel']
labelAll = excavator
rigthLabel = excavator
smallsize = 48
smallsize2 = 71
smallNum = 0
largeNum = 0
count1 = 0
count2 = 0
count3 = 0
for txtfile in glob.glob(originfile):
    print('jsonfile:', txtfile)
    imagePath = txtfile.replace('txt', 'jpg')
    imageBaseName = os.path.basename(imagePath)
    img = Image.open(imagePath)
    W, H = img.size
    boxes = []
    labelme_formate = {
        "version": "4.2.9",
        "flags": {},
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": imageBaseName,
        "imageHeight": H,
        "imageWidth": W
    }
    labelme_formate['imageData'] = None
    shapes = []
    with open(txtfile, 'r') as f:
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
            s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
            points = [
                [xmin, ymin],
                [xmax, ymax]
            ]
            s['points'] = points
            shapes.append(s)
    labelme_formate['shapes'] = shapes
    json.dump(labelme_formate, open(saveJoson + imageBaseName.replace('jpg', 'json'), 'w'), ensure_ascii=False, indent=2)
    shutil.copy(imagePath, saveJoson)


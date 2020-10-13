import json
#import pandas as pd
import os
import numpy as np
import glob
import shutil

jsonSavePathSmallObject = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin_small_0_71/'
jsonSavePathLargeObject = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin_large_48_96/'
originJson = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin/*.json'
if os.path.exists(jsonSavePathSmallObject):
    shutil.rmtree(jsonSavePathSmallObject)
os.mkdir(jsonSavePathSmallObject)

if os.path.exists(jsonSavePathLargeObject):
    shutil.rmtree(jsonSavePathLargeObject)
os.mkdir(jsonSavePathLargeObject)

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
for jsonfile in glob.glob(originJson):
    print('jsonfile:', jsonfile)
    imagePath = jsonfile.replace('json', 'jpg')
    imageBaseName = os.path.basename(imagePath)
    jsondict = json.load(open(jsonfile, 'r'), encoding='gb2312')
    shapes = jsondict['shapes']
    W = float(jsondict['imageWidth'])
    H = float(jsondict['imageHeight'])
    labelme_formate_small = {
        "version": "4.2.9",
        "flags": {},
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": imageBaseName,
        "imageHeight": H,
        "imageWidth": W
    }
    labelme_formate_small['imageData'] = None
    shapes_small = []

    labelme_formate_large = {
        "version": "4.2.9",
        "flags": {},
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": imageBaseName,
        "imageHeight": H,
        "imageWidth": W
    }
    labelme_formate_large['imageData'] = None
    shapes_large = []
    for shape in shapes:
        label = shape['label']
        assert label in labelAll, label
        if label not in rigthLabel:
            continue
        label_id = -1
        if label in excavator:
            count1 = count1 + 1
            label_id = 0
        if label in truck:
            count2 = count2 + 1
            label_id = 1
        if label in wheel:
            count3 = count3 + 1
            label_id = 2
        points = np.array(shape['points'])
        xmin = float((min(points[:, 0])))
        xmax = float((max(points[:, 0])))
        ymin = float((min(points[:, 1])))
        ymax = float((max(points[:, 1])))
        imgW = xmax - xmin
        imgH = ymax - ymin
        ppoint = [
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]]
        ]
        if imgH < smallsize2:
            s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
            s['points'] = ppoint
            shapes_small.append(s)
            smallNum = smallNum + 1
        if imgH > smallsize:
            s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
            s['points'] = ppoint
            shapes_large.append(s)
            largeNum = largeNum + 1
    if len(shapes_small) :
        labelme_formate_small['shapes'] = shapes_small
        json.dump(labelme_formate_small, open(jsonSavePathSmallObject + imageBaseName.replace('jpg', 'json'), 'w'),
                  ensure_ascii=False, indent=2)
        shutil.copy(imagePath, jsonSavePathSmallObject + imageBaseName)
    if len(shapes_large):
        labelme_formate_large['shapes'] = shapes_large
        json.dump(labelme_formate_large, open(jsonSavePathLargeObject + imageBaseName.replace('jpg', 'json'), 'w'), ensure_ascii=False, indent=2)
        shutil.copy(imagePath, jsonSavePathLargeObject + imageBaseName)
print('smallObjNum:', smallNum)
print('largeObjNum:', largeNum)
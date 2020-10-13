import json
#import pandas as pd
import os
import numpy as np
import glob
import shutil
import random
import cv2

def allObj(shapes):
    allObjPoint = []
    for shape in shapes:
        points = np.array(shape['points'])
        xmin = float((min(points[:, 0])))
        xmax = float((max(points[:, 0])))
        ymin = float((min(points[:, 1])))
        ymax = float((max(points[:, 1])))
        ppoint = [
            [xmin, ymin],
            [xmax, ymax]
        ]
        allObjPoint.append(ppoint)
    return allObjPoint

def cal_iou_change(rec_1, rec_2):
    area_1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])
    area_2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])

    cross_left = max(rec_1[1], rec_2[1])
    cross_right = min(rec_1[3], rec_2[3])
    cross_top = max(rec_1[0], rec_2[0])
    cross_bottom = min(rec_1[2], rec_2[2])
    cross_area = (cross_right - cross_left) * (cross_bottom - cross_top)  # n
    total_area = area_1 + area_2 - cross_area  # u

    if cross_left >= cross_right or cross_top >= cross_bottom:
        return 0
    else:
        return round(cross_area / float(area_2), 2)


def cal_insert(point1, point2):
    ppoint1 = point1[0] + point1[1]
    ppoint2 = point2[0] + point2[1]
    IOU = cal_iou_change(ppoint1, ppoint2)
    if point2[0][0]>=point1[0][0] and point2[0][1]>=point1[0][1] and point2[1][0]<=point1[1][0] and point2[1][1]<=point1[1][1] or IOU > 0.6:
        flag = 1
    else:
        flag = 0
    return flag



jsonSavePathSmallObject = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/person_lone/'
originJson = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/zt_done/*.json'
if os.path.exists(jsonSavePathSmallObject):
    shutil.rmtree(jsonSavePathSmallObject)
os.mkdir(jsonSavePathSmallObject)
excavator = ['sea_person', 'earth_person', 'person']
car = ['car']
truck = ['truck']
other = ['other']
wheel = ['wheel']
labelAll = excavator
rigthLabel = excavator
count1 = 0
count2 = 0
count3 = 0
randRatioX = 0
randRatioY = 0
for jsonfile in glob.glob(originJson):
    print('jsonfile:', jsonfile)
    imagePath = jsonfile.replace('json', 'jpg')
    originImage = cv2.imread(imagePath)
    imageBaseName = os.path.basename(imagePath)[:-4]
    jsondict = json.load(open(jsonfile, 'r'), encoding='gb2312')
    shapes = jsondict['shapes']
    W = float(jsondict['imageWidth'])
    H = float(jsondict['imageHeight'])
    count1=0
    allObjPoint = allObj(shapes)
    for shape in shapes:
        shapes_small = []
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
        xmin = float(max((min(points[:, 0])), 0))
        xmax = float(min((max(points[:, 0])), W))
        ymin = float(max((min(points[:, 1])),0))
        ymax = float(min((max(points[:, 1])), H))
        imgH = int(ymax - ymin)
        imgW = int(xmax - xmin)
        saveImageName = imageBaseName+'_' + str(imgH) + '_' + str(imgW) + '_' +str(count1)+'.jpg'
        saveimg = originImage[int(ymin):int(ymax), int(xmin):int(xmax)]
        cv2.imwrite(jsonSavePathSmallObject+saveImageName, saveimg)
    print(count1)
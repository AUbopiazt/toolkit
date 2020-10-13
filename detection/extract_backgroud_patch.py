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

def cal_iou(rec_1, rec_2):
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
        return round(cross_area / float(total_area), 2)


def cal_insert(point1, point2):
    ppoint1 = point1[0] + point1[1]
    ppoint2 = point2[0] + point2[1]
    IOU = cal_iou(ppoint1, ppoint2)
    if point2[0][0]>=point1[0][0] and point2[0][1]>=point1[0][1] and point2[1][0]<=point1[1][0] and point2[1][1]<=point1[1][1] or IOU > 0.01:
        flag = 1
    else:
        flag = 0
    return flag



jsonSavePathSmallObject = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/backgroud/'
originJson = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin/*.json'
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
backgroudSaveNum = 50
for jsonfile in glob.glob(originJson):
    print('jsonfile:', jsonfile)
    imagePath = jsonfile.replace('json', 'jpg')
    originImage = cv2.imread(imagePath)
    imageBaseName = os.path.basename(imagePath)[:-4]
    jsondict = json.load(open(jsonfile, 'r'), encoding='gb2312')
    shapes = jsondict['shapes']
    W = float(jsondict['imageWidth'])
    H = float(jsondict['imageHeight'])
    randRatioX = W
    randRatioY = H
    expandW = 20 #classfication 45    detect  96
    expandH = 20 #classfication 45    detect  96
    count1=0
    allObjPoint = allObj(shapes)
    while count1<backgroudSaveNum:
        xmin_rand = randRatioX * random.uniform(0, 1)
        ymin_rand = randRatioY * random.uniform(0, 1)
        crop_W = expandW * random.uniform(0.2, 5)
        crop_H = expandH * random.uniform(0.2, 5)
        xmax_rand = xmin_rand + crop_W
        ymax_rand = ymin_rand + crop_H
        if xmax_rand>W or ymax_rand>H:
            continue
        backgroudBox = [[xmin_rand, ymin_rand],
                        [xmax_rand, ymax_rand]]
        flag = 0
        for objBox in allObjPoint:
            flag = cal_insert(backgroudBox, objBox)
            if flag == 1:
                break
        if flag == 1:
            continue
        saveImageName = imageBaseName + '_backgourd_' + str(int(crop_H)) + '_' +str(int(crop_W)) + '_' + str(count1) + '.jpg'
        txtname = saveImageName.replace('jpg', 'txt')
        saveimg = originImage[int(ymin_rand):int(ymax_rand), int(xmin_rand):int(xmax_rand)]
        cv2.imwrite(jsonSavePathSmallObject + saveImageName, saveimg)
        backgroudtxt = open(jsonSavePathSmallObject+txtname, 'w')
        backgroudtxt.close()
        count1 = count1+1

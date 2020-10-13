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
    if point2[0][0]>=point1[0][0] and point2[0][1]>=point1[0][1] and point2[1][0]<=point1[1][0] and point2[1][1]<=point1[1][1] or IOU > 0.3:
        flag = 1
    else:
        flag = 0
    return flag



jsonSavePathSmallObject = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/crop/'
originJson = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin/*.json'
# if os.path.exists(jsonSavePathSmallObject):
#     shutil.rmtree(jsonSavePathSmallObject)
# os.mkdir(jsonSavePathSmallObject)
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
        #assert label in labelAll, label
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

        originPoint = [
            [xmin, ymin],
            [xmax, ymax]
        ]
        # if ymin > H / 2.5:
        #     continue
        imgW = xmax - xmin
        imgH = ymax - ymin
        randRatioX = imgW
        randRatioY = imgH*0.8
        expandx = randRatioX * random.uniform(1, 2)
        expandy = randRatioY * random.uniform(0.5, 1)
        expandxmax = randRatioX * random.uniform(1, 2)
        expandymax = randRatioY * random.uniform(0.5, 1)
        cropx_min = max(0, xmin - expandx)
        cropy_min = max(0, ymin - expandy)
        cropx_max = min(W, xmax + expandxmax)
        cropy_max = min(H, ymax + expandymax)


        boxOrigin = [
            [cropx_min, cropy_min],
            [cropx_max, cropy_max]
        ]

        objectxmin = expandx if cropx_min!=0 else xmin
        objectymin = expandy if cropy_min!=0 else ymin

        saveJson = imageBaseName+'_'+str(count1)+'.json'
        saveImageName = saveJson.replace('json', 'jpg')
        saveimg = originImage[int(cropy_min):int(cropy_max), int(cropx_min):int(cropx_max)]
        ppoint = [
            [objectxmin, objectymin],
            [objectxmin+imgW, objectymin+imgH]
        ]
        s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
        s['points'] = ppoint
        shapes_small.append(s)
        objIndex = 0
        for obj in allObjPoint:
            # if objIndex==count1:
            #     objIndex=objIndex+1
            #     continue
            flag = cal_insert(boxOrigin, obj)
            if flag==1:
                span_x_min = originPoint[0][0] - obj[0][0]
                span_y_min = originPoint[0][1] - obj[0][1]
                span_x_max = originPoint[1][0] - obj[1][0]
                span_y_max = originPoint[1][1] - obj[1][1]
                if span_y_max==0 and span_x_max==0 and span_y_min==0 and span_x_min==0:
                    continue
                new_x_min = max(ppoint[0][0]-span_x_min, 0)
                new_y_min = max(ppoint[0][1]-span_y_min, 0)
                new_x_max = min(ppoint[1][0]-span_x_max, cropx_max-cropx_min)
                new_y_max = min(ppoint[1][1]-span_y_max, cropy_max-cropy_min)

                for_new_point = [
                    [new_x_min, new_y_min],
                    [new_x_max, new_y_max]
                ]
                s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
                s['points'] = for_new_point
                shapes_small.append(s)
            objIndex = objIndex + 1
        labelme_formate_small = {
            "version": "4.2.9",
            "flags": {},
            "lineColor": [0, 255, 0, 128],
            "fillColor": [255, 0, 0, 128],
            "imagePath": saveImageName,
            "imageHeight": int(cropy_max-cropy_min),
            "imageWidth": int(cropx_max-cropx_min)
        }
        labelme_formate_small['imageData'] = None
        labelme_formate_small['shapes'] = shapes_small
        json.dump(labelme_formate_small, open(jsonSavePathSmallObject + saveJson, 'w'),
                  ensure_ascii=False, indent=2)
        cv2.imwrite(jsonSavePathSmallObject+saveImageName, saveimg)
    print(count1)
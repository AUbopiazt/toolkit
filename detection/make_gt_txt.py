'''
labelme(json) to txt
The txt contain all label
'''
import json
import glob
import os
import numpy as np

gt = open('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/excavator_gt.txt', 'w')
imgPath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/0827testxmu/*.jpg'
#imgPath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3/caffe-yolov3-master/person/*.jpg'
validLabel = ['excavator', 'truck', 'other', 'car', 'wheel', 'person']
excavator = ['person', 'truck', 'wheel']
#excavator = ['person']
labelID = ['1', '2', '3']
label2ID = dict(zip(excavator, labelID))
for img in glob.glob(imgPath):
    imgName = img.split('/')[-1]
    jsonPath = img.replace('jpg', 'json')
    if os.path.exists(jsonPath):
        jsonfile = open(jsonPath, 'r')
        jsonLoad = json.load(jsonfile, encoding='gb2312')
        bbox = str()
        for items in jsonLoad['shapes']:
            label = items['label']
            assert label in validLabel, label
            if label not in excavator:
                continue

            points = np.array(items['points'])
            xmin = int(min(points[:, 0]))
            xmax = int(max(points[:, 0]))
            ymin = int(min(points[:, 1]))
            ymax = int(max(points[:, 1]))
            bbox += ' {label} {xmin} {ymin} {xmax} {ymax}'.format(label=str(label2ID[label]), xmin=str(xmin),
                                                                 ymin=str(ymin), xmax=str(xmax), ymax=str(ymax))
        line = imgName + bbox
        gt.write(line + '\n')
        jsonfile.close()
    else:
        line = imgName
        gt.write(line + '\n')
gt.close()




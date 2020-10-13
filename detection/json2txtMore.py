import glob
import json
import numpy as np
import shutil
import os
#excavator = ['excavator', 'cavator', 'pile_driver', 'push_bench', 'execavator']
excavator = ['sea_person', 'earth_person', 'person']
car = ['car']
truck = ['truck']
other = ['other']
wheel = ['wheel']
labelAll = excavator
rigthLabel = excavator
jsonPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin/*.json'
savepath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/layerout/'
# if os.path.exists(savepath):
#     shutil.rmtree(savepath)
# os.mkdir(savepath)
ss = len(glob.glob(jsonPath))
count1 = 0
count2 = 0
count3 = 0
for jsonfile in glob.glob(jsonPath):
    txtfile = jsonfile.replace('json', 'txt').replace('origin', 'layerout')
    imgfile = jsonfile.replace('json', 'jpg').replace('origin', 'layerout')
    print(imgfile)
    txt = open(txtfile, 'w')
    jsondict = json.load(open(jsonfile, 'r'), encoding='gb2312')
    shapes = jsondict['shapes']
    W = float(jsondict['imageWidth'])
    H = float(jsondict['imageHeight'])
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
        centerx = (xmin + xmax) / 2
        centery = (ymin + ymax) / 2

        imgW = imgW / W
        imgH = imgH / H
        centerx = centerx / W
        centery = centery / H
        txt.write(str(label_id) + ' ' + str(centerx) + ' ' + str(centery) + ' ' + str(imgW) + ' ' + str(imgH) + '\n')
    txt.close()
    shutil.copy(jsonfile.replace('json', 'jpg'), imgfile)
print('{},{},{}'.format(count1, count2, count3))



import glob
import json
import numpy as np
import shutil
import os
import xml.etree.ElementTree as ET

def parse_annotation(path):
    assert(os.path.exists(path)), \
        'Annotation: {} does not exist'.format(path)
    tree = ET.parse(path)
    objs = tree.findall('object')
    boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        box = [x1, y1, x2, y2]
        cls = obj.find('name').text.lower().lower()
        #difficult = int(obj.find('difficult').text) == 1
        # difficult = False
        # if FILTER_DIFFICULT:
        #     if not difficult:
        #         boxes.append({'cls': cls, 'box': box})
        # else:
        boxes.append({'cls': cls, 'box': box})
    return boxes


#excavator = ['excavator', 'cavator', 'pile_driver', 'push_bench', 'execavator']
excavator = ['sea_person', 'earth_person', 'person']
car = ['car']
truck = ['truck']
other = ['other']
wheel = ['wheel']
labelAll = excavator
rigthLabel = excavator
xmlPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/voc/Annotations/*.xml'
savepath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/darknet_train/'
if os.path.exists(savepath):
    shutil.rmtree(savepath)
os.mkdir(savepath)
#ss = len(glob.glob(jsonPath))
count1 = 0
count2 = 0
count3 = 0
for xmlfile in glob.glob(xmlPath):
    txtfile = xmlfile.replace('xml', 'txt').replace('voc/Annotations', 'darknet_train')
    imgfile = xmlfile.replace('xml', 'jpg').replace('voc/Annotations', 'darknet_train')
    print(imgfile)
    txt = open(txtfile, 'w')
    tree = ET.parse(xmlfile)
    objs = tree.findall('object')
    imgsize = tree.find('size')
    W = float(imgsize.find('width').text)
    H = float(imgsize.find('height').text)
    boxes = []
    for obj in objs:
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        box = [x1, y1, x2, y2]
        label = obj.find('name').text.lower().lower()
        #difficult = int(obj.find('difficult').text) == 1
        # difficult = False
        # if FILTER_DIFFICULT:
        #     if not difficult:
        #         boxes.append({'cls': cls, 'box': box})
        # else:
        boxes.append({'label': label, 'box': box})
    for box in boxes:
        label = box['label']
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
        points = np.array(box['box'])
        xmin = float((min(points[0], points[2])))
        xmax = float((max(points[0], points[2])))
        ymin = float((min(points[1], points[3])))
        ymax = float((max(points[1], points[3])))
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
    shutil.copy(xmlfile.replace('xml', 'jpg').replace('Annotations', 'JPEGImages'), imgfile)
print('{},{},{}'.format(count1, count2, count3))



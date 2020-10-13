# -*- coding: utf-8 -*
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
import random
#from sklearn.model_selection import train_test_split

# 1.标签路径
labelme_path = "/media/aubopiazt/AA6CE0AF6CE07789/dataFormate/waterMassTrain/excavatorTruckWheel/excavator/"  # 原始labelme标注数据路径
saved_path = "/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/VOCdevkit/excavator/"  # 保存路径

# 2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")

# 3.获取待处理文件
files = glob(labelme_path + "*.json")
# kk = files[954:958]
# print(kk[1])
files = [i.split("/")[-1].split(".json")[0] for i in files]
excavator = ['excavator', 'cavator', 'pile_driver', 'push_bench']
car = ['car']
truck = ['truck']
other = ['other']
wheel = ['wheel']
labelAll = excavator + truck + other + wheel + car
# 4.读取标注信息并写入 xml
for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    #print('name:', json_filename)
    #json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    #json_file = json.load(open(json_filename, "r", encoding="gb2312"))
    json_file = json.load(open(json_filename, mode='r'), encoding='gb2312')
    height, width, channels = cv2.imread(labelme_path + json_file_ + ".jpg").shape
    with codecs.open(saved_path + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>wave</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = int(min(points[:, 0]))
            xmax = int(max(points[:, 0]))
            ymin = int(min(points[:, 1]))
            ymax = int(max(points[:, 1]))
            label = multi["label"]
            assert label in labelAll
            if label in excavator:
                label = 'excavator'
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + label + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>0</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename, xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')

# 5.复制图片到 VOC2007/JPEGImages/下
image_files = glob(labelme_path + "*.jpg")
print("copy image files to VOC007/JPEGImages/")
for image in image_files:
    shutil.copy(image, saved_path + "JPEGImages/")

# 6.split files for txt
xmlfilepath = saved_path + 'Annotations/'  # change xml path
saveBasePath = saved_path + 'ImageSets'  # change base path
trainval_percent = 0.8  # adjust trainval percentage
train_percent = 0.8  # adjust train percentage
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

print("train and val size", tv)
print("traub suze", tr)
ftrainval = open(os.path.join(saveBasePath, 'Main/trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'Main/test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'Main/train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'Main/val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
#
# txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath + '/trainval.txt', 'w')
# ftest = open(txtsavepath + '/test.txt', 'w')
# ftrain = open(txtsavepath + '/train.txt', 'w')
# fval = open(txtsavepath + '/val.txt', 'w')
# total_files = glob(saved_path + "Annotations/*.xml")
# total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
# # test_filepath = ""
# for file in total_files:
#     ftrainval.write(file + "\n")
# # test
# for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
# # split
# train_files, val_files = train_test_split(total_files, test_size=0.1, random_state=42)
# # train
# for file in train_files:
#     ftrain.write(file + "\n")
# # val
# for file in val_files:
#     fval.write(file + "\n")
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()
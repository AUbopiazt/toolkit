import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
from kmeans import kmeans, avg_iou


#ANNOTATIONS_PATH = "/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/datasets/excavator/excavator/EffectiveData/data/voc/train.txt"#/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/Gaussian_YOLOv3/darknet/datatococo/darknet_excavator.txt
xmlpath = '/media/aubopiazt/reid/zt/objectDet/personOne/'
CLUSTERS = 9
anchor_box_path ='/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/anchor_box.txt'
anchor_box = open(anchor_box_path, 'w')
def takeKey(elem):
    return elem[0]

def load_dataset(path):
    dataset = []
    #for xml_file in glob.glob("{}/*xml".format(path)):
    for xml_file in glob.glob(path+'/*.txt'):
        img_file = xml_file.replace('.txt', '.jpg')
        img = Image.open(img_file)
        imgW, imgH = img.size
        with open(xml_file, 'r') as f:
            txtread = f.readlines()
            for lines in txtread:
                txtText = lines.rstrip().split(' ')
                xmin = float(txtText[2])
                ymin = float(txtText[3])
                xmax = float(txtText[4])
                ymax = float(txtText[5])
                norm_w = abs(xmax - xmin) / imgW
                norm_h = abs(ymax - ymin) / imgH
                objW = norm_w * 1
                objH = norm_h * 1
                dataset.append([objW, objH])

    return np.array(dataset)


def load_dataset_txt(path):
    dataset = []
    f1 = open(path)
    for line in f1.readlines():
        txtPath = line.strip('\n').replace('.jpg','.txt')
        print(txtPath)
        f2 = open(txtPath)
        for textLine in f2.readlines():
            textLine = textLine.strip('\n')
            w, h = textLine.strip().split(' ')[3:]
            w, h = map(float, (w, h))
            dataset.append([w, h])
    return np.array(dataset)


data = load_dataset(xmlpath)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
out[:, 0] = out[:, 0] * 256  #512  #w
out[:, 1] = out[:, 1] * 192  #320  #h
print("Boxes:\n {}".format(out))

index = np.argsort(out, axis=0)
#w
# sort_out = out[index[:, 0]]

#h
sort_out = out[index[:, 1]]
print("Sort:\n {}".format(sort_out))
print('***************************')
mobilenetAnchor = sort_out.reshape(1, -1).squeeze()
anchor_box.write('anchors=')
for i in mobilenetAnchor:
    darknet = str(round(i, 7)) + ','
    anchor_box.write(darknet)
anchor_box.write('\n'*3)
for i in mobilenetAnchor:
    mobilenet = 'biases:' + str(round(i, 7)) + '\n'
    anchor_box.write(mobilenet)
anchor_box.close()

ss = list(out*512)
kk = []
for i in ss:
    kk.append(i)
print(kk)

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

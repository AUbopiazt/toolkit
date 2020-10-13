import json
import shutil

cocoJsonPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/coco2017/person_keypoints_train2017.json'
cocoJson = open(cocoJsonPath, 'r')
cocoJsonText = json.load(cocoJson, encoding='gb2312')
csvPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/coco2017/person_keypoints_train2017.csv'
csvLabels = open(csvPath, 'w')
annotations = cocoJsonText['annotations']
images = cocoJsonText['images']
categories = cocoJsonText['categories']
id2imageName = {}
id2imgsize = {}
id2category = {}
#NextImage = 0
for image in images:
    imgid = image['id']
    imageName = image['file_name'].split('/')[-1]
    id2imageName[imgid] = imageName
    width = image['width']
    height = image['height']
    whList = [width, height]
    id2imgsize[imgid] = whList
for category in categories:
    catgoryId = category['id']
    labelName = category['name']
    id2category[catgoryId] = labelName
for annotation in annotations:
    imageId = annotation['image_id']
    isCrowd = annotation['iscrowd']
    numKeypoints = annotation['num_keypoints']
    #ignore = annotation['ignore']
    #uncertain = annotation['uncertain']
    #logo = annotation['logo']
    categoryId = annotation['category_id']
    bbox = annotation['bbox']
    xmin = float(bbox[0])
    ymin = float(bbox[1])
    xmax = xmin + float(bbox[2])
    ymax = ymin + float(bbox[3])
    imageNames = id2imageName[imageId]
    if imageNames == '000000006233.jpg':
        kk = 0
    label = id2category[categoryId]
    width = id2imgsize[imageId][0]
    height = id2imgsize[imageId][1]
    if isCrowd == 1 or numKeypoints < 3:
        print('iscrowd = ', isCrowd)
        print('numkeypoint = ', numKeypoints)
        continue
    else:
        csvLabels.write(imageNames + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + label+ ',' + str(width) + ',' + str(height) + '\n')
cocoJson.close()
csvLabels.close()
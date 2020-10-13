import json
import pandas as pd
import os
import numpy as np
jsonSavePath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/coco2017/coco2017_train_person/'
csvPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/coco2017/person_keypoints_train2017.csv'
annotations = pd.read_csv(csvPath, header=None).values
total_csv_annotations = {}
count = 0
for annotatin in annotations:
    # imgname = annotatin[0]
    # xmin = annotatin[1]
    # ymin = annotatin[2]
    # xmax = annotatin[3]
    # ymax = annotatin[4]
    # label = annotatin[5]
    # width = annotatin[6]
    # height = annotatin[7]
    key = annotatin[0].split(os.sep)[-1]
    value = np.array([annotatin[1:]])
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
    else:
        total_csv_annotations[key] = value
for key, value in total_csv_annotations.items():
    count = count + 1
    width = value[0][5]
    height = value[0][6]
    labelme_formate = {
        "version": "4.2.9",
        "flags": {},
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": key,
        "imageHeight": height,
        "imageWidth": width
    }
    labelme_formate['imageData'] = None
    shapes = []
    for shape in value:
        label = shape[4]
        s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
        points = [
            [shape[0], shape[1]],
            [shape[2], shape[3]]
        ]
        s['points'] = points
        shapes.append(s)
    labelme_formate['shapes'] = shapes
    json.dump(labelme_formate, open(jsonSavePath + key.replace('jpg', 'json'), 'w'), ensure_ascii=False, indent=2)
    # if count>6000:
    #     break
    print(count)
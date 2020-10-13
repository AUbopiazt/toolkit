import json
#import pandas as pd
import os
import numpy as np
import glob
import shutil
from PIL import Image


trackfilepath = '/media/aubopiazt/reid/zt/GigaVision_modify/video_annos/'
saveRoot = '/media/aubopiazt/reid/zt/GigaVision_modify/'
for root, dirs, files in os.walk(trackfilepath):
    for dir in dirs:
        sencePath = root + '/' + dir
        trackFile = sencePath + '/tracks.json'
        nameFile = sencePath + '/seqinfo.json'
        saveJoson = saveRoot + dir + '/'
        if not os.path.exists(saveJoson):
            os.mkdir(saveJoson)
        trackJsonDic = json.load(open(trackFile, 'r'), encoding='gb2312')
        nameJosonDic = json.load(open(nameFile, 'r'), encoding='gb2312')
        imgNames = nameJosonDic['imUrls']
        W = nameJosonDic['imWidth'] / 10
        H = nameJosonDic['imHeight'] / 10
        frame = 1
        label = 'person'
        occlusion = ['serious hide', 'normal', 'hide', 'disappear', '']

        for imgName in imgNames:
            labelme_formate = {
                "version": "4.2.9",
                "flags": {},
                "lineColor": [0, 255, 0, 128],
                "fillColor": [255, 0, 0, 128],
                "imagePath": imgName,
                "imageHeight": H,
                "imageWidth": W
            }
            labelme_formate['imageData'] = None
            shapes = []
            jsonName = imgName.replace('jpg', 'json')
            for trackid in trackJsonDic:
                for trackidFrame in trackid['frames']:
                    if trackidFrame['frame id'] == frame:
                        assert trackidFrame['occlusion'] in occlusion, trackidFrame['occlusion']
                        # if trackidFrame['occlusion'] == 'serious hide' or trackidFrame['occlusion'] == 'hide' or trackidFrame['occlusion'] == 'disappear':
                        if trackidFrame['occlusion'] != 'normal':
                            continue
                        bbox = trackidFrame['rect']
                        xmin = bbox['tl']['x'] * W
                        ymin = bbox['tl']['y'] * H
                        xmax = bbox['br']['x'] * W
                        ymax = bbox['br']['y'] * H
                        points = [[xmin, ymin],
                                  [xmax, ymax]]
                        s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
                        s['points'] = points
                        shapes.append(s)
                    else:
                        continue
            labelme_formate['shapes'] = shapes
            json.dump(labelme_formate, open(saveJoson + jsonName, 'w'), ensure_ascii=False, indent=2)
            frame = frame + 1


# trackFile = '/media/aubopiazt/reid/zt/GigaVision/video_annos/07_University_Campus/tracks.json'
# nameFile = '/media/aubopiazt/reid/zt/GigaVision/video_annos/07_University_Campus/seqinfo.json'
# saveJoson = '/media/aubopiazt/reid/zt/GigaVision/07_modify/'
# trackJsonDic = json.load(open(trackFile, 'r'), encoding='gb2312')
# nameJosonDic = json.load(open(nameFile, 'r'), encoding='gb2312')
# imgNames = nameJosonDic['imUrls']
# W = nameJosonDic['imWidth'] / 10
# H = nameJosonDic['imHeight'] / 10
# frame = 1
# label = 'person'
# occlusion = ['serious hide', 'normal', 'hide', 'disappear', '']
#
# for imgName in imgNames:
#     labelme_formate = {
#         "version": "4.2.9",
#         "flags": {},
#         "lineColor": [0, 255, 0, 128],
#         "fillColor": [255, 0, 0, 128],
#         "imagePath": imgName,
#         "imageHeight": H,
#         "imageWidth": W
#     }
#     labelme_formate['imageData'] = None
#     shapes = []
#     jsonName = imgName.replace('jpg', 'json')
#     for trackid in trackJsonDic:
#         for trackidFrame in trackid['frames']:
#             if trackidFrame['frame id'] == frame:
#                 assert trackidFrame['occlusion'] in occlusion, trackidFrame['occlusion']
#                 #if trackidFrame['occlusion'] == 'serious hide' or trackidFrame['occlusion'] == 'hide' or trackidFrame['occlusion'] == 'disappear':
#                 if trackidFrame['occlusion'] == 'serious hide' or trackidFrame['occlusion'] == 'hide' or trackidFrame[
#                     'occlusion'] == 'disappear':
#                     continue
#                 bbox = trackidFrame['rect']
#                 xmin = bbox['tl']['x'] * W
#                 ymin = bbox['tl']['y'] * H
#                 xmax = bbox['br']['x'] * W
#                 ymax = bbox['br']['y'] * H
#                 points = [[xmin, ymin],
#                           [xmax, ymax]]
#                 s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
#                 s['points'] = points
#                 shapes.append(s)
#             else:
#                 continue
#     labelme_formate['shapes'] = shapes
#     json.dump(labelme_formate, open(saveJoson + jsonName, 'w'), ensure_ascii=False, indent=2)
#     frame = frame + 1
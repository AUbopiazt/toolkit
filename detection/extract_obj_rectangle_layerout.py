# -*- coding: utf-8 -*-
import numpy as np
from json import load
import cv2 as cv
import json
import glob
import cv2
import os
import random

def get_v(a, b):
    mx = (a > b) and a or b
    mi = (a <= b) and a or b
    return mi / float(mx)

class MergeBlock:
    def __init__(self):
        self.ori = list()
        self.merge = list()
        self.candidates = list()
        self.mask = np.array([])
        self.output_size = 640
        self.delta_area = 0
        self.index = list()
        self.W = 0
        self.H = 0
        self.area_weight = 0.3
        self.wh_ratio = 1 - self.area_weight

    def set_output_size(self, output_size):
        self.output_size = output_size

    def init(self, input_):
        self.candidates = [[0, 0]]
        self.delta_area = 0
        self.W = 0
        self.H = 0
        self.index = list()
        self.ori = input_
        self.merge = list()
        self.mask = np.zeros((self.output_size, self.output_size), dtype=np.uint8)

    def deploy(self, r):
        scores = list()

        for i in range(len(self.candidates)):
            curr_mask = np.zeros(self.mask.shape, np.uint8)
            p = self.candidates[i]
            curr_mask[p[1]: p[1]+r[3], p[0]: p[0]+r[2]] = 1
            temp = cv.bitwise_and(self.mask, curr_mask)
            if np.sum(temp) > 0:
                scores.append(0)
            else:
                w = p[0] + r[2]
                h = p[1] + r[3]
                w = (self.W > w) and self.W or w
                h = (self.H > h) and self.H or h
                i_area = float(w*h) / (self.delta_area + r[2]*r[3])
                i_wh = get_v(w, h)
                score = i_wh / i_area
                scores.append(score)
        ind = np.argsort(-np.array(scores))[0]
        p = self.candidates[ind]
        r[0] = p[0]
        r[1] = p[1]
        w = p[0] + r[2]
        h = p[1] + r[3]
        self.W = (self.W > w) and self.W or w
        self.H = (self.H > h) and self.H or h

        self.delta_area += r[2]*r[3]
        self.update_candidates(r, ind)
        return r

    def update_candidates(self, rec, ind):
        del self.candidates[ind]
        p_lb = [rec[0], rec[1]+rec[3]]
        p_ru = [rec[0]+rec[2], rec[1]]
        self.candidates.append(p_lb)
        self.candidates.append(p_ru)

    def re_layout(self, input_):
        self.init(input_)
        max_size_len = np.array([], dtype=np.int)
        for r in self.ori:
            w = r[2]
            h = r[3]
            max_size_len = np.append(max_size_len, [(w > h) and w or h])
        self.index = np.argsort(-max_size_len)
        for i in range(self.index.size):
            rec = self.ori[self.index[i]]
            m_rec = [rec[0], rec[1], rec[2], rec[3]]
            if i==0:
                m_rec[0] = 0
                m_rec[1] = 0
                self.W = m_rec[2]
                self.H = m_rec[3]
                self.delta_area += m_rec[2]*m_rec[3]
                self.update_candidates(m_rec, 0)
            else:
                m_rec = self.deploy(m_rec)

            self.mask[m_rec[1]:m_rec[1]+m_rec[3], m_rec[0]:m_rec[0]+m_rec[2]] = 1
            self.merge.append(m_rec)

    def extract(self, mat, save_path):
        if self.H == 0 or self.W == 0:
            return
        print('[W H area_v w_h] {} ,{} ,{}, {}'.format(self.W, self.H, self.delta_area / (self.W * self.H),
                                                       get_v(self.W, self.H)))
        temp = np.zeros((self.H, self.W, 3), np.uint8)
        for i in range(len(self.merge)):
            x, y, w, h = self.merge[i]
            img = mat[self.index[i]]
            #x0, y0, w0, h0 = self.ori[self.index[i]]
            #temp[y:y+h, x:x+w, :] = mat[y0:y0+h0, x0:x0+w0, :]
            temp[y:y + h, x:x + w, :] = img
        cv.imwrite(save_path, temp)

    def saveJson(self, bbox, save_path):
        if self.H == 0 or self.W == 0:
            return
        shapes_small = []
        labelme_formate_small = {
            "version": "4.2.9",
            "flags": {},
            "lineColor": [0, 255, 0, 128],
            "fillColor": [255, 0, 0, 128],
            "imagePath": os.path.basename(save_path).replace('json', 'jpg'),
            "imageHeight": self.H,
            "imageWidth": self.W
        }
        labelme_formate_small['imageData'] = None
        for i in range(len(self.merge)):

            for k in range(len(bbox[self.index[i]])):
                bbox[self.index[i]][k][0][0] = self.merge[i][0] + bbox[self.index[i]][k][0][0]
                bbox[self.index[i]][k][1][0] = self.merge[i][0] + bbox[self.index[i]][k][1][0]

                bbox[self.index[i]][k][0][1] = self.merge[i][1] + bbox[self.index[i]][k][0][1]
                bbox[self.index[i]][k][1][1] = self.merge[i][1] + bbox[self.index[i]][k][1][1]

                #print('w:{}, h:{}'.format(bbox[self.index[i]][k][0][1]-bbox[self.index[i]][k][0][0], bbox[self.index[i]][k][1][1]-bbox[self.index[i]][k][1][0]))
                s = {"label": 'person', "line_color": None, "fill_color": None, "shape_type": "rectangle"}
                s['points'] = bbox[self.index[i]][k]
                shapes_small.append(s)
        labelme_formate_small['shapes'] = shapes_small
        json.dump(labelme_formate_small, open(save_path, 'w'),
                  ensure_ascii=False, indent=2)



def load_labelme(file_path):
    input_ = list()
    with open(file_path, 'r', encoding='utf-8') as fd:
        dct = load(fd)
    ind = 0
    for i in dct['shapes']:
        if i['label'] == 'person' and i['shape_type'] == 'rectangle':
            rec = np.array(i['points'], dtype=np.int)
            x = np.min(rec[:, 0])
            y = np.min(rec[:, 1])
            w = np.max(rec[:, 0]) - x
            h = np.max(rec[:, 1]) - y
            if w * h > 6400:
                continue
            input_.append([x, y, w, h])
            ind += 1

    return input_


class ExtractObj:
    def __init__(self):
        self.excavator = ['sea_person', 'earth_person', 'person']
        self.car = ['car']
        self.truck = ['truck']
        self.other = ['other']
        self.wheel = ['wheel']
        self.labelAll = self.excavator
        self.rigthLabel = self.excavator
        self.count1 = 0
        self.count2 = 0
        self.count3 = 0
        self.randRatioX = 0
        self.randRatioY = 0

    def cal_iou_change(self, rec_1, rec_2):
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

    def cal_insert(self, point1, point2):
        ppoint1 = point1[0] + point1[1]
        ppoint2 = point2[0] + point2[1]
        IOU = self.cal_iou_change(ppoint1, ppoint2)
        if point2[0][0] >= point1[0][0] and point2[0][1] >= point1[0][1] and point2[1][0] <= point1[1][0] and point2[1][
            1] <= point1[1][1] or IOU > 0.3:
            flag = 1
        else:
            flag = 0
        return flag

    def allObj(self, shapes):
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

    def extractObj(self, jsonfile):

        print('jsonfile:', jsonfile)
        imagePath = jsonfile.replace('json', 'jpg')
        originImage = cv2.imread(imagePath)
        imageBaseName = os.path.basename(imagePath)[:-4]
        jsondict = json.load(open(jsonfile, 'r'), encoding='gb2312')
        shapes = jsondict['shapes']
        W = float(jsondict['imageWidth'])
        H = float(jsondict['imageHeight'])
        self.count1 = 0
        crop_img_vector = []
        crop_img_obj_for_all_point = []
        allObjPoint = self.allObj(shapes)
        for shape in shapes:
            crop_img_obj_point = []
            shapes_small = []
            label = shape['label']
            # assert label in labelAll, label
            if label not in self.rigthLabel:
                continue
            label_id = -1
            if label in self.excavator:
                self.count1 = self.count1 + 1
                label_id = 0
            if label in self.truck:
                self.count2 = self.count2 + 1
                label_id = 1
            if label in self.wheel:
                self.count3 = self.count3 + 1
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
            randRatioX = imgW * 0.4
            randRatioY = imgH * 0.3
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

            objectxmin = expandx if cropx_min != 0 else xmin
            objectymin = expandy if cropy_min != 0 else ymin

            #saveJson = imageBaseName + '_' + str(self.count1) + '.json'
            #saveImageName = saveJson.replace('json', 'jpg')
            crop_img = originImage[int(cropy_min):int(cropy_max), int(cropx_min):int(cropx_max)]
            crop_img_vector.append(crop_img)
            ppoint = [
                [objectxmin, objectymin],
                [objectxmin + imgW, objectymin + imgH]
            ]
            crop_img_obj_point.append(ppoint)
            for obj in allObjPoint:
                # if objIndex==count1:
                #     objIndex=objIndex+1
                #     continue
                flag = self.cal_insert(boxOrigin, obj)
                if flag == 1:
                    span_x_min = originPoint[0][0] - obj[0][0]
                    span_y_min = originPoint[0][1] - obj[0][1]
                    span_x_max = originPoint[1][0] - obj[1][0]
                    span_y_max = originPoint[1][1] - obj[1][1]
                    if span_y_max == 0 and span_x_max == 0 and span_y_min == 0 and span_x_min == 0:
                        continue
                    new_x_min = max(ppoint[0][0] - span_x_min, 0)
                    new_y_min = max(ppoint[0][1] - span_y_min, 0)
                    new_x_max = min(ppoint[1][0] - span_x_max, cropx_max - cropx_min)
                    new_y_max = min(ppoint[1][1] - span_y_max, cropy_max - cropy_min)

                    for_new_point = [
                        [new_x_min, new_y_min],
                        [new_x_max, new_y_max]
                    ]
                    crop_img_obj_point.append(for_new_point)
            crop_img_obj_for_all_point.append(crop_img_obj_point)
        return crop_img_obj_for_all_point, crop_img_vector
    def extractBackground(self, jsonfile, savePath, backgroudSaveNum):

        #print('jsonfile:', jsonfile)
        imagePath = jsonfile.replace('json', 'jpg')
        originImage = cv2.imread(imagePath)
        imageBaseName = os.path.basename(imagePath)[:-4]
        jsondict = json.load(open(jsonfile, 'r'), encoding='gb2312')
        shapes = jsondict['shapes']
        W = float(jsondict['imageWidth'])
        H = float(jsondict['imageHeight'])
        self.count1 = 0
        randRatioX = W
        randRatioY = H
        expandW = 20  # classfication 45    detect  96
        expandH = 20  # classfication 45    detect  96
        count1 = 0
        allObjPoint = self.allObj(shapes)
        backgoroudImageVector = list()
        while count1 < backgroudSaveNum:
            xmin_rand = randRatioX * random.uniform(0, 1)
            ymin_rand = randRatioY * random.uniform(0, 1)
            crop_W = expandW * random.uniform(0.2, 5)
            crop_H = expandH * random.uniform(0.2, 5)
            xmax_rand = xmin_rand + crop_W
            ymax_rand = ymin_rand + crop_H
            if xmax_rand > W or ymax_rand > H:
                continue
            backgroudBox = [[xmin_rand, ymin_rand],
                            [xmax_rand, ymax_rand]]
            flag = 0
            for objBox in allObjPoint:
                flag = self.cal_insert(backgroudBox, objBox)
                if flag == 1:
                    break
            if flag == 1:
                continue
            # img_savepath = savePath + imageBaseName + '_backgroud_' + str(count1) + '.jpg'
            # json_savepath = img_savepath.replace('jpg', 'json')
            # labelme_formate_small = {
            #     "version": "4.2.9",
            #     "flags": {},
            #     "lineColor": [0, 255, 0, 128],
            #     "fillColor": [255, 0, 0, 128],
            #     "imagePath": imageBaseName + '_backgroud_' + str(count1) + '.jpg',
            #     "imageHeight": int(ymax_rand-ymin_rand),
            #     "imageWidth": int(xmax_rand-xmin_rand)
            # }
            # labelme_formate_small['imageData'] = None
            # json.dump(labelme_formate_small, open(json_savepath, 'w'),
            #           ensure_ascii=False, indent=2)
            backgouroudImg = originImage[int(ymin_rand):int(ymax_rand), int(xmin_rand):int(xmax_rand)]
            backgoroudImageVector.append(backgouroudImg)
            # cv.imwrite(img_savepath, backgouroudImg)
            count1 = count1 + 1
        return backgoroudImageVector




def test_extract_obj():
    label = 'person'
    jsonSavePathSmallObject = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/toolkit/detection/test/'
    originJson = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/toolkit/detection/*.json'
    # if os.path.exists(jsonSavePathSmallObject):
    #     shutil.rmtree(jsonSavePathSmallObject)
    # os.mkdir(jsonSavePathSmallObject)
    extr = ExtractObj()
    for jsonfile in glob.glob(originJson):
        crop_obj_point, crop_img = extr.extractObj(jsonfile)
        for i in range(len(crop_img)):
            shapes_small = []
            cv2.imwrite(jsonSavePathSmallObject + str(i) + '.jpg', crop_img[i])
            labelme_formate_small = {
                "version": "4.2.9",
                "flags": {},
                "lineColor": [0, 255, 0, 128],
                "fillColor": [255, 0, 0, 128],
                "imagePath": str(i)+'.jpg',
                "imageHeight": crop_img[i].shape[0],
                "imageWidth": crop_img[i].shape[1]
            }
            labelme_formate_small['imageData'] = None
            for k in range(len(crop_obj_point[i])):
                s = {"label": label, "line_color": None, "fill_color": None, "shape_type": "rectangle"}
                s['points'] = crop_obj_point[i][k]
                shapes_small.append(s)
            labelme_formate_small['shapes'] = shapes_small
            json.dump(labelme_formate_small, open(jsonSavePathSmallObject + str(i) + '.json', 'w'),
                      ensure_ascii=False, indent=2)

def load_point_img_list(cropImg, cropPoint):
    input_ = list()
    crop_img_input = list()
    crop_obj_point_input = list()
    numImg = (len(cropImg) < 1000) and len(cropImg) or int(len(cropImg)*0.5)

    for i in range(numImg):
        h = cropImg[i].shape[0]
        w = cropImg[i].shape[1]
        # if h * w >= 6000:
        #     continue
        input_.append([0, 0, w, h])
        crop_img_input.append(cropImg[i])
        crop_obj_point_input.append(cropPoint[i])
    return input_, crop_img_input, crop_obj_point_input


def main():
    jsonSavePathSmallObject = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/data/layerout/'
    originJson = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/data/gys/*.json'
    # if os.path.exists(jsonSavePathSmallObject):
    #     shutil.rmtree(jsonSavePathSmallObject)
    # os.mkdir(jsonSavePathSmallObject)
    extr = ExtractObj()
    mb = MergeBlock()
    for jsonfile in glob.glob(originJson):
        baseJsonName = os.path.basename(jsonfile)
        baseImgName = baseJsonName.replace('json', 'jpg')
        crop_obj_point, crop_img = extr.extractObj(jsonfile)
        if len(crop_img)==0:
            continue

        backgourdImage = extr.extractBackground(jsonfile, jsonSavePathSmallObject, len(crop_img))
        backgourdPoint = [[]] * len(crop_img)
        crop_and_backgourd_img = crop_img + backgourdImage
        crop_and_backgourd_point = crop_obj_point + backgourdPoint
        c = list(zip(crop_and_backgourd_img, crop_and_backgourd_point))
        random.shuffle(c)
        crop_and_backgourd_img, crop_and_backgourd_point = zip(*c)
        # randnum = random.randint(0, 100)
        # random.seed(randnum)
        # random.shuffle(crop_and_backgourd_img)
        # random.seed(randnum)
        # random.shuffle(crop_and_backgourd_point)
        input_, crop_img_input, crop_obj_point_input = load_point_img_list(crop_and_backgourd_img, crop_and_backgourd_point)
        #input_, crop_img_input, crop_obj_point_input = load_point_img_list(crop_img, crop_obj_point)
        mb.re_layout(input_)
        mb.extract(crop_img_input,jsonSavePathSmallObject+baseImgName)
        mb.saveJson(crop_obj_point_input, jsonSavePathSmallObject+baseJsonName)









    # mb = MergeBlock()
    # input_ = load_labelme('SEQ_01_119.json')
    # mat = cv.imread('SEQ_01_119.jpg')
    # mb.re_layout(input_)
    # mb.extract(mat, 'temp.jpg')


if __name__ == '__main__':
    main()
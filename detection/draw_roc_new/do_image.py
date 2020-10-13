# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import os
save_img_dir = "/home/xuefei/work/FOD/ROC/400/gt/ori/"
result_img_dir = "/home/xuefei/work/FOD/ROC/400/result/"
obj_dir = save_img_dir + "obj/"
back_dir = save_img_dir + "back/"


def add_label_to_image(mat, label, color):
    for i_label in label:
        label_len = len(i_label)
        if label_len == 4:
            color = (255, 0, 0)
            cv.rectangle(mat, (int(i_label[0]), int(i_label[1])), (int(i_label[2]), int(i_label[3])), color, 2)
        elif label_len == 5:
            color = (0, 0, 255)
            cv.rectangle(mat, (int(i_label[0]), int(i_label[1])), (int(i_label[2]), int(i_label[3])), color, 3)
        else:
            return 0
        #cv.putText(mat, txt, (i_label[0], i_label[1]), cv.FONT_HERSHEY_COMPLEX, 1, color, 1)
    return mat

##############################################################
def CalDistance(mixBoxone, mixBoxtwo):
    flag = False
    flag1 = False
    flag2 = False
    disx = abs((mixBoxone[0] + mixBoxone[2]) * 0.5 - (mixBoxtwo[0] + mixBoxtwo[2]) * 0.5)
    whx = (mixBoxone[2] - mixBoxone[0]) * 0.5 + (mixBoxtwo[2] - mixBoxtwo[0])*0.5
    if disx-whx<25:
        flag1 = True
    disy = (abs((mixBoxone[1] + mixBoxone[3]) * 0.5 - (mixBoxtwo[1] + mixBoxtwo[3]) * 0.5))
    why = (mixBoxone[3] - mixBoxone[1]) * 0.5 + (mixBoxtwo[3] - mixBoxtwo[1]) * 0.5
    if disy-why<25:
        flag2 = True
    if flag1 and flag2:
        flag = True
    return flag

def CalIoU(cx1,cy1,cx2,cy2,gx1,gy1,gx2,gy2):
    iou = 0
    carea = (cx2 - cx1) * (cy2 - cy1)
    garea = (gx2 - gx1) * (gy2 - gy1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    iou = area / ((carea + garea - area)+0.0001)
    return iou
def maxBox(img_rectangle, mixBox,dst_name):
    result = []
    list_num = 0
    i = 0
    for a in range(len(mixBox)):

        box_1 = mixBox[a]
        ######################################### get label and score from net ######################################################
        boxW = float(box_1[2] - box_1[0])
        boxH = float(box_1[3] - box_1[1])
        rate_small = 0.35
        rate_hide_0 = 0.1
        rate_hide_1 = 1.2
        rate_big = 0.2
        mixbox0 = 0
        mixbox1 = 0
        mixbox2 = 0
        mixbox3 = 0
        scoreSmall = 0.0
        scoreBig = 0.0
        label = 0  # obj
        if boxW * boxH < 30 * 30 and (  # boxW * boxH > 10 * 10 and
                boxW / boxH > 0.5 and boxW / boxH < 2):  # small
            mixbox0 = (int)(max(0, float(box_1[0]) - boxW * rate_small))
            mixbox1 = (int)(max(0, float(box_1[1]) - boxH * rate_small))
            mixbox2 = (int)(min(img_rectangle.shape[1], float(box_1[2]) + boxW * rate_small))
            mixbox3 = (int)(min(img_rectangle.shape[0], float(box_1[3]) + boxH * rate_small))
            curRectImg = img_rectangle[mixbox1:mixbox3, mixbox0:mixbox2]
            if 0 in curRectImg.shape:
                continue
            cv.imwrite(dst_name + '_' + str(i) + '_' + '.jpg', curRectImg)
            i += 1
        elif boxW * boxH >= 30 * 30 and (boxW / boxH > 0.5 and boxW / boxH < 2):  # big
            mixbox0 = (int)(max(0, float(box_1[0]) - boxW * rate_big))
            mixbox1 = (int)(max(0, float(box_1[1]) - boxH * rate_big))
            mixbox2 = (int)(min(img_rectangle.shape[1], float(box_1[2]) + boxW * rate_big))
            mixbox3 = (int)(min(img_rectangle.shape[0], float(box_1[3]) + boxH * rate_big))
            curRectImg = img_rectangle[mixbox1:mixbox3, mixbox0:mixbox2]
            if 0 in curRectImg.shape:
                continue
            cv.imwrite(dst_name + '_' + str(i) + '_' + '.jpg', curRectImg)
            i += 1
        elif boxW > 5 and boxH > 5:  # hide boxW * boxH > 10 * 10 and
            if boxW > boxH:
                mixbox0 = (int)(max(0, float(box_1[0]) - boxW * rate_hide_0))
                mixbox1 = (int)(max(0, float(box_1[1]) - boxH * rate_hide_1))
                mixbox2 = (int)(min(img_rectangle.shape[1], float(box_1[2]) + boxW * rate_hide_0))
                mixbox3 = (int)(min(img_rectangle.shape[0], float(box_1[3]) + boxH * rate_hide_1))
            elif boxW <= boxH:
                mixbox0 = (int)(max(0, float(box_1[0]) - boxW * rate_hide_1))
                mixbox1 = (int)(max(0, float(box_1[1]) - boxH * rate_hide_0))
                mixbox2 = (int)(min(img_rectangle.shape[1], float(box_1[2]) + boxW * rate_hide_1))
                mixbox3 = (int)(min(img_rectangle.shape[0], float(box_1[3]) + boxH * rate_hide_0))
            curRectImg = img_rectangle[mixbox1:mixbox3, mixbox0:mixbox2]
            if (mixbox2 - mixbox0) * (mixbox3 - mixbox1) >= 30 * 30:
                if 0 in curRectImg.shape:
                    continue
                cv.imwrite(dst_name + '_' + str(i) + '_' + '.jpg', curRectImg)
                i += 1
            else:
                if 0 in curRectImg.shape:
                    continue
                cv.imwrite(dst_name + '_' + str(i) + '_' + '.jpg', curRectImg)
                i += 1

def MixBox(iter,annotation):
    mixBox = []
    calBoxID = []
    for a in range(iter):
        data_1 = annotation[a * 5 :(a + 1) * 5]
        if len(data_1) == 0:
            continue
        else:
            # print data_1
            box_1 = (int(float(data_1[0])), int(float(data_1[1])), int(float(data_1[2])), int(float(data_1[3])))
        isContinue = False
        for b in range(len(calBoxID)):
            if a == calBoxID[b]:
                isContinue = True
                break
        if isContinue:
            continue
        cross = False
        calBoxID.append(a)
        for b in range(len(mixBox)):
            iou = CalIoU(box_1[0], box_1[1], box_1[2], box_1[3], mixBox[b][0], mixBox[b][1], mixBox[b][2], mixBox[b][3])
            flag = CalDistance(box_1, mixBox[b])
            if iou > 0 or flag:  # 25 pixel

                cross = True
                box1_0 = min(box_1[0], mixBox[b][0])
                box1_1 = min(box_1[1], mixBox[b][1])
                box1_2 = max(box_1[2], mixBox[b][2])
                box1_3 = max(box_1[3], mixBox[b][3])
                mixBox.append((box1_0, box1_1, box1_2, box1_3))
        if cross == False:
            mixBox.append(box_1)
        for b1 in range(0, len(mixBox) - 1):
            for b2 in range(b1 + 1, len(mixBox)):
                iou = CalIoU(mixBox[b1][0], mixBox[b1][1], mixBox[b1][2], mixBox[b1][3], mixBox[b2][0], mixBox[b2][1],
                             mixBox[b2][2], mixBox[b2][3])
                flag = CalDistance(mixBox[b1], mixBox[b2])
                if iou > 0 or flag:  # 25 pixel
                    box1_0 = min(mixBox[b1][0], mixBox[b2][0])
                    box1_1 = min(mixBox[b1][1], mixBox[b2][1])
                    box1_2 = max(mixBox[b1][2], mixBox[b2][2])
                    box1_3 = max(mixBox[b1][3], mixBox[b2][3])
                    mixBox[b2] = (box1_0, box1_1, box1_2, box1_3)
                    mixBox[b1] = (10001, 10001, 1, 1)
                    break
        for b in range(len(mixBox) - 1, -1, -1):
            if mixBox[b][0] > 10000:
                del (mixBox[b])
    return mixBox



def copy_aug_box_from_image(mat, label, dst_name):
    # print label
    annotation = label.flatten()
    #annotation = label.strip().split(' ')
    long = len(annotation)
    iter = int((long) / 5)
    mixBox = MixBox(iter, annotation)
    maxBox(mat, mixBox,dst_name)


    # for i_label in label:
    #     label_len = len(i_label)
    #     mixBox = MixBox(iter, label)
    #     # print int(i_label[0]), int(i_label[2]), int(i_label[1]), int(i_label[3]), dst_name + '_' + str(i) + '_' + 'jpg'
    #     sub_image = mat[int(i_label[1]): int(i_label[3]), int(i_label[0]): int(i_label[2]), :]
    #     if float(i_label[4]) >= 0.3:
    #         cv.imwrite(dst_name + '_' + str(i) + '_' + '.jpg', sub_image)
    #         i += 1
def copy_box_from_image(mat, label, dst_name):
    for i_label in label:
        label_len = len(i_label)
        # mixBox = MixBox(iter, label)
        # print int(i_label[0]), int(i_label[2]), int(i_label[1]), int(i_label[3]), dst_name + '_' + str(i) + '_' + 'jpg'
        sub_image = mat[int(i_label[1]): int(i_label[3]), int(i_label[0]): int(i_label[2]), :]
        if float(i_label[4]) >= 0.3:
            cv.imwrite(dst_name + '_' + str(i_label[0])+'_'+ str(i_label[1])+'_' +str(i_label[2])+'_'+str(i_label[3])+'_' + '.jpg', sub_image)

def do_for_single_image(image_dir, image_name, dt, gt, iou_score_array):
    # print(dt)

    iou_max = np.max(iou_score_array[:, 1])
    if iou_max > 0:
        ind = np.where(iou_score_array[:, 1] == 0)
        ind_true  = np.where(iou_score_array[:, 1] > 0)
        ind_len = len(ind[0])
        ind_true_len = len(ind_true[0])
    else:
        ind = range(len(dt))
        ind_len = len(dt)
        ind_true = ()
        ind_true_len = 0
    if ind_len > 0:
        # print image_name

        image_path = os.path.join(image_dir, image_name)
        mat = cv.imread(image_path)
        dst_name = back_dir  + image_name[:-4]
        # print dst_name
        # copy_aug_box_from_image(mat, np.array(dt)[ind], dst_name)
        #copy_box_from_image(mat, np.array(dt)[ind], dst_name)
        # mat = add_label_to_image(mat, np.array(dt)[ind], (0, 0, 255))  # 错误的检出
        # print image_name
        # cv.imwrite(result_img_dir+image_name, mat)
        # cv.imshow(image_name, cv.resize(mat, (1800, 900)))

        # cv.waitKey(0)
    if ind_true_len > 0:
        image_path = os.path.join(image_dir, image_name)
        mat = cv.imread(image_path)
        dst_name = obj_dir + image_name[:-4]
        # print dst_name
        #copy_box_from_image(mat, np.array(dt)[ind_true], dst_name)
        # mat = add_label_to_image(mat, np.array(dt)[ind], (0, 0, 255))  # 错误的检出
        # print image_name
        # cv.imwrite(result_img_dir + image_name, mat)
        # cv.imwrite(dst_name + '_' + str(i) + '_' + '.jpg', sub_image)
        # cv.imshow(image_name, cv.resize(mat, (1800, 900)))
        # cv.waitKey(0)
        # copy_box_from_image(mat, np.array(dt)[ind_true], dst_name)


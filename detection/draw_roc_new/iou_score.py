# -*- coding:utf-8 -*-
import numpy as np
from do_image import do_for_single_image


def cal_iou(rec_1, rec_2):
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
        return round(cross_area / float(total_area), 2)


def get_iou_score_lst_from_iou_array(iou_array, score_array):
    iou_score_array = np.vstack((score_array, np.max(iou_array, axis=0))).T
    return iou_score_array


def cal_iou_array(dt_lst, gt_lst):
    dt_num = len(dt_lst)
    gt_num = len(gt_lst)
    iou_array = np.zeros((gt_num, dt_num))
    i = 0
    # 计算IOU
    for gt in gt_lst:
        rec_1 = gt[: 4]
        j = 0
        for dt in dt_lst:
            rec_2 = dt[: 4]
            iou = cal_iou(rec_1, rec_2)
            iou_array[i, j] = iou
            j += 1
        i += 1
    return iou_array


def cal_iou_array_for_single_image(dt_lst, gt_lst):
    dt_num = len(dt_lst)
    gt_num = len(gt_lst)
    iou_array = np.zeros((gt_num, dt_num))
    i = 0
    # 计算IOU
    for gt in gt_lst:
        rec_1 = gt[: 4]
        j = 0
        for dt in dt_lst:
            rec_2 = dt[: 4]
            iou = cal_iou(rec_1, rec_2)
            iou_array[i, j] = iou
            j += 1
        i += 1
    # print iou_array
    j = 0
    # IOU的二次处理(找到GT框对应的最大的IOU，同时score最大的DT，作为唯一的检出，其余置为0)
    for gt_line in iou_array:
        max_v = np.max(gt_line)
        if max_v > 0:
            score_array = dt_lst[:, 4]
            # index of dt who's iou is same
            max_iou_ind = np.where(gt_line == max_v)
            # index of dt who's score is same, iou is same
            ind_score_array = dt_lst[max_iou_ind, 4]
            max_score_ind = np.where(score_array == np.max(ind_score_array))
            # when iou and score is same, we pick the first dt in the sub as the TP.
            ind = np.intersect1d(max_score_ind, max_iou_ind)    # 交集
            zero_iou_ind = np.setdiff1d(range(dt_num), ind[0])  # 补集
            iou_array[j, zero_iou_ind] = 0
        j += 1
    # print iou_array
    return iou_array, dt_lst[:, 4]


def iou_thres(iou_array, iou_thr):
    iou_array[:, 1] = (iou_array[:, 1] >= iou_thr)
    return iou_array


def get_iou_score_lst(dt, gt, image_dir):
    set_iou_score_array = np.empty((0, 2))
    for key in dt:
        if gt.get(key, 'None') != 'None':
            # print key
            if len(gt[key]) > 0 and len(dt[key]) > 0:
                iou_array, score = cal_iou_array_for_single_image(np.array(dt[key]), np.array(gt[key]))
            elif len(gt[key]) == 0 and len(dt[key]) > 0:
                score = np.array(dt[key])[:, 4]
                iou_array = np.zeros((1, len(score)))
            else:
                continue
            iou_score_array = get_iou_score_lst_from_iou_array(iou_array, score)
            do_for_single_image(image_dir, key, dt[key], gt[key], iou_score_array)
            set_iou_score_array = np.append(set_iou_score_array, iou_score_array, axis=0)
        else:
            print('\033[1;33m  [WARN] \033[0mThere has no gt for image[{}]'.format(key))

    set_iou_score_array = set_iou_score_array[np.argsort(-set_iou_score_array[:, 0])]
    return set_iou_score_array



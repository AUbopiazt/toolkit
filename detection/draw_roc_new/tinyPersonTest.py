# -*- coding:utf-8 -*-
import numpy as np
from label_load import get_label_from_txt
from iou_score import iou_thres, get_iou_score_lst, cal_iou_array
from indicator import print_index, draw_roc, get_indicator
from util_ import get_cfg_from_json
import prettytable as pt

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

def print_index(dt_smallObject, dt_mediumObject, dt_bigObject, gt_smallObject, gt_mediumObject, gt_bigObject, smallObject):
    tb = pt.PrettyTable()
    tb.field_names = [' ', 'small', 'medium', 'big', 'sum', 'smallObject_size']
    tb.add_row(['dt', dt_smallObject, dt_mediumObject, dt_bigObject, dt_smallObject+dt_mediumObject+dt_bigObject, smallObject])
    tb.add_row(['gt', gt_smallObject, gt_mediumObject, gt_bigObject, gt_smallObject+gt_mediumObject+gt_bigObject, smallObject])
    tb.add_row(['ratio', float(dt_smallObject)/gt_smallObject, float(dt_mediumObject)/gt_mediumObject, float(dt_bigObject)/gt_bigObject, float(dt_smallObject+dt_mediumObject+dt_bigObject)/(gt_smallObject+gt_mediumObject+gt_bigObject), smallObject])
    print tb

def get_iou(dt, gt, image_dir):
    set_iou_score_array = np.empty((0, 2))
    dt_smallObject = 0
    dt_mediumObject = 0
    dt_bigObject = 0
    gt_smallObject = 0
    gt_mediumObject = 0
    gt_bigObject = 0
    smallObject = 48
    mediumObject = 96

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
            #do_for_single_image(image_dir, key, dt[key], gt[key], iou_score_array)
            set_iou_score_array = np.append(set_iou_score_array, iou_score_array, axis=0)
            dt_array = np.array(dt[key])
            gt_array = np.array(gt[key])
            i = 0
            for ious in iou_array:
                kk = np.argwhere(ious > 0.5)
                if len(kk)>0:
                    ind = int(np.argwhere(ious > 0.5))
                    dt_bbox = dt_array[ind]
                    dt_height = dt_bbox[3] - dt_bbox[1]


                    if dt_height < smallObject:
                        dt_smallObject = dt_smallObject + 1
                    elif dt_height < mediumObject:
                        dt_mediumObject = dt_mediumObject + 1
                    else:
                        dt_bigObject = dt_bigObject + 1

                gt_bbox = gt_array[i]
                gt_height = gt_bbox[3] - gt_bbox[1]
                if gt_height < smallObject:
                    gt_smallObject = gt_smallObject + 1
                elif gt_height < mediumObject:
                    gt_mediumObject = gt_mediumObject + 1
                else:
                    gt_bigObject = gt_bigObject + 1

                i = i + 1


        else:
            print('\033[1;33m  [WARN] \033[0mThere has no gt for image[{}]'.format(key))

    print_index(dt_smallObject, dt_mediumObject, dt_bigObject, gt_smallObject, gt_mediumObject, gt_bigObject, smallObject)
    set_iou_score_array = set_iou_score_array[np.argsort(-set_iou_score_array[:, 0])]
    return set_iou_score_array


def solver(dt, dt_count, gt, gt_count, iou_thr, images_dir):
    iou_score_array = get_iou(dt, gt, images_dir)

    kk = 0

    # iou_score_array = iou_thres(iou_score_array, iou_thr)
    # score_ind = np.argwhere(iou_score_array[:, 0] <= 0.6)
    # fp_lst, recall_lst, precision_lst, auc, mAP = get_indicator(iou_score_array, gt_count, dt_count)
    # print('auc= %f, mAP = %f ', auc, mAP)
    # # max_recall_ind = recall_lst.index(np.max(recall_lst))
    #
    # error_num = np.max(fp_lst[:score_ind[0][0]])
    # tp_dt_num = gt_count * recall_lst[score_ind[0][0]]
    # loss_num = gt_count - tp_dt_num
    # print_index(dt_count, gt_count, error_num, loss_num, tp_dt_num, iou_score_array[score_ind[0][0], 0])
    # draw_roc(fp_lst, recall_lst, auc)


def main():
    # test_lst = ['1218', 'data3_imagefortest100', 'oldobj', 'phoneobj', 'toolobj', 'setsizeobj', 'real']
    test_lst = ['excavator']
    for test in test_lst:
        cfg = get_cfg_from_json('test_cfg.json', test)
        dt, dt_count = get_label_from_txt(cfg['dt']['txt_loc'], cfg['dt']['label_len'],
                                          cfg['dt']['label_format'], cfg['class_lst'], cfg['labelIndex'])
        gt, gt_count = get_label_from_txt(cfg['gt']['txt_loc'], cfg['gt']['label_len'],
                                          cfg['gt']['label_format'], cfg['class_lst'], cfg['labelIndex'])
        print ( dt_count, gt_count)
        #iou_array = cal_iou_array(dt,gt)
        #for d in dt:
        solver(dt, dt_count, gt, gt_count, cfg['iou_thr'], cfg['images_dir'])


if __name__ == '__main__':
    main()



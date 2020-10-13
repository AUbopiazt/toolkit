# -*- coding:utf-8 -*-
import numpy as np
from label_load import get_label_from_txt
from iou_score import iou_thres, get_iou_score_lst, cal_iou_array
from indicator import print_index, draw_roc, get_indicator
from util_ import get_cfg_from_json


def solver(dt, dt_count, gt, gt_count, iou_thr, images_dir):
    iou_score_array = get_iou_score_lst(dt, gt, images_dir)
    iou_score_array = iou_thres(iou_score_array, iou_thr)
    score_ind = np.argwhere(iou_score_array[:, 0] <= 0.6)
    fp_lst, recall_lst, precision_lst, auc, mAP = get_indicator(iou_score_array, gt_count, dt_count)
    print('auc= %f, mAP = %f ', auc, mAP)
    # max_recall_ind = recall_lst.index(np.max(recall_lst))

    error_num = np.max(fp_lst[:score_ind[0][0]])
    tp_dt_num = gt_count * recall_lst[score_ind[0][0]]
    loss_num = gt_count - tp_dt_num
    print_index(dt_count, gt_count, error_num, loss_num, tp_dt_num, iou_score_array[score_ind[0][0], 0])
    draw_roc(fp_lst, recall_lst, auc)


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



import numpy as np
import matplotlib.pyplot as plt
import prettytable as pt
from progressBar import progressBar
import time


def get_indicator(iou_score_array, gt_count, dt_count):
    dt_count = len(iou_score_array)
    auc = 0; mAP = 0
    fp_lst = []; recall_lst = []; precision_lst = []

    bar = progressBar('Indicator')
    for num in bar.tlist(range(dt_count)):
        arr = iou_score_array[:(num + 1), 1]
        tp = np.sum(arr)
        fp = np.sum(arr == 0)
        recall = tp / gt_count
        precision = tp / (tp + fp)
        fp_lst.append(fp)
        recall_lst.append(recall)
        precision_lst.append(precision)

        auc = auc + recall
        mAP = mAP + precision
    time.sleep(1)
    # print num
    print [iou_score_array[num, 0], fp, round(recall, 6)]
    auc = round(auc / dt_count, 3)
    mAP = round(mAP * max(recall_lst) / dt_count, 3)
    return fp_lst, recall_lst, precision_lst, auc, mAP


def draw_roc(fp_lst, recall_lst, auc):
    plt.figure()
    plt.title('ROC')
    plt.xlabel('False Positives')
    plt.ylabel('True Positive rate')
    plt.ylim(0, 1)
    #plt.plot(fp_lst, recall_lst, label='AUC: ' + str(auc))
    plt.plot(fp_lst, recall_lst)
    plt.legend()
    plt.show()
    # plt.savefig(save_name)


def print_index(dt_count, gt_count, error_num, loss_num, tp_dt_num, score):
    tb = pt.PrettyTable()
    tb.field_names = ['gt', 'dt', 'score', 'error', 'loss', 'tp_dt_num']
    tb.add_row([gt_count, dt_count, score, error_num, loss_num, tp_dt_num])
    tb.add_row([gt_count, dt_count, score, error_num, loss_num, tp_dt_num])
    print tb
# -*- coding:utf-8 -*-


def get_label_from_txt(txt_name, label_len, label_format, class_lst, labelIndex):
    format_lst = label_format.split('_')
    if label_len == 5:
        new_format_lst = ['x1', 'y1', 'x2', 'y2', 'label']
    else:
        new_format_lst = ['x1', 'y1', 'x2', 'y2', 'label', 'score']
    ind_lst = label_format_transform(format_lst, new_format_lst)
    labels_dct = dict()
    count = 0
    with open(txt_name, 'r') as fd:
        for line in fd.readlines():                             # 逐图片
            ind = line.find('.png')
            if ind==-1:
                ind = line.find('.jpg')
                image_name = line[:ind]+'.jpg'
            else:
                image_name = line[:ind] + '.png'
            tags = line[ind:].strip('\r\n').split(' ')
            label_count = (len(tags) - 1) / label_len
            label_lst = list()
            for i in range(label_count):                        # 逐框
                label_ = label_transform(tags[i * label_len + 1: (i + 1) * label_len + 1], ind_lst)
                # label = label_[4]
                # if label_len == 5:
                #     label = labelIndex[int(label_[4])]
                # if label_len == 6:
                label = labelIndex[int(label_[4])]
                if label in class_lst:
                    box = [int(j) for j in label_[: 4]]
                    if label_len == 5:
                        d_label = box
                    else:
                        d_label = box + [float(label_[5])]
                    label_lst.append(d_label)
                    count += 1
            labels_dct[image_name] = label_lst
        return labels_dct, count


def label_transform(label, ind_lst):
    label_ = list()
    for ind in ind_lst:
        label_.append(label[ind])
    return label_


def label_format_transform(old_format_lst, new_format_lst):
    format_lst = list()
    for item in new_format_lst:
        format_lst.append(old_format_lst.index(item))
    return format_lst

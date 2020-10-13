# -*- coding: UTF-8 -*
import os
import cv2 as cv
import time

caffe_root = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/MobileNet-YOLO-master/'
# os.chdir(caffe_root)
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

show_img_flag = False
save_img_flag = True
save_img_path = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/test_result/'
# net_path = '/home/goalin/project/caffeversion/MobileNet-SSD/no_bn.prototxt'
# caffemodel_path = '/home/goalin/project/caffeversion/MobileNet-SSD/no_bn.caffemodel'
#excavator
# model_root = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/ztModel/'
# model_type = 'blur_HD_all/'
# model_time = '20200730/'
# model_file_name = 'mobilenet_deploy_voc.prototxt'
# model_weight_name = 'mobilenet_yolov3_coco_320_excavator_blur_HD_all_1class_voc_iter_100000.caffemodel'

#person
model_root = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/ztModel/'
model_type = 'person/'#'blur_HD_all/'
model_time = '20200826/'
model_file_name = 'mobilenet_deploy_class1_coco.prototxt'#'mobilenet_deploy_voc.prototxt'
model_weight_name = 'mobilenet_yolov3_voc_person_iter_96000.caffemodel'#'mobilenet_yolov3_coco_320_excavator_blur_HD_all_1class_voc_iter_100000.caffemodel'

net_path = model_root + model_type + model_time + model_file_name
caffemodel_path = model_root + model_type + model_time + model_weight_name

# dt_txt = open('/home/xuefei/work/detect_roc/yolov3_caffe/excavator_lhl/add_camera/dt-5000-outdoor.txt', 'w')

# dt_txt = open('/home/goalin/project/Datas/waterMassData/lmdb/test/excavatorData_gt_dt.txt', 'w')
dt_txt = open('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/excavator_dt.txt', 'w')

# image_dir = '/home/goalin/project/Datas/waterMassData/lmdb/test/'
image_dir = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/alltest/'
# txt_path = '/home/goalin/project/Datas/waterMassData/lmdb/test/excavatorData_gt.txt'
txt_path = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/excavator_gt.txt'

classModels = '/home/goalin/project/caffeversion/MobileNet-YOLO/peleenet/YNExcavator_2020_03_7_iter_108000.caffemodel'
classprototxt = '/home/goalin/project/caffeversion/MobileNet-YOLO/peleenet/peleenet.prototxt'


def classModel():
    #caffe.set_mode_cpu()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(classprototxt, classModels, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    return transformer, net


def model_load():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(net_path, caffemodel_path, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))

    #mu = np.array([127.5, 127.5, 127.5])   # [1, 1, 1]
    mu = np.array([1, 1, 1])  #
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 0.007843)  # )
    # net.blobs['data'].reshape(1,  # batch size
    #                           3,  # 3-channel (BGR) images
    #                           512, 512)

    return transformer, net


def forward_img(transformer, net, mat):
    w = mat.shape[1]
    h = mat.shape[0]
    transformed_image = transformer.preprocess('data', mat)
    net.blobs['data'].data[...] = transformed_image
    start = time.clock()
    net.forward()
    t = round((time.clock() - start) * 1000, 2)  # ms   processor
    # image_name label score x_min y_min x_max y_max ...
    res, score_lst = result_filter(net.blobs['detection_out'].data[0].flatten(), w, h)

    return res, t


def classForwardImg(transformer, net, mat, res_lst):
    h, w, c = mat.shape
    mm = 0
    for kk in range(len(res_lst)):
        kk = mm
        mm = mm + 1
        scoreD = res_lst[kk][:1]
        rect = res_lst[kk][2:]
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        ws = x2 - x1
        hs = y2 - y1
        if ws * hs < 50 * 50 or scoreD <= 0.2:
            mm = mm - 1
            continue
        addw = 0.1 * ws
        addh = 0.1 * hs
        x1 = int(x1 - addw * 0.5)
        y1 = int(y1 - addh * 0.5)
        x2 = int(x2 + addw * 0.5)
        y2 = int(y2 + addh * 0.5)
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h
        img = mat[y1:y2, x1:x2]
        # cv.imwrite(save_img_path + '1.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 95])
        im = cv.resize(img, (196, 196))
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        out = net.forward()
        prob = net.blobs['prob'].data[0].flatten()
        print(prob)
        order = prob.argsort()[-1]  # small-big
        if order == 0:
            res_lst.remove(res_lst[kk])
            mm = mm - 1
            continue
        print(order)
    return res_lst


def normalization(num, lower_, upper_):
    if num < lower_:
        return lower_
    elif num > upper_:
        return upper_
    else:
        return num


def result_filter(lst, w, h):
    res_lst = []
    score_lst = []
    num = len(lst) / 7
    obj_num = 0
    for i in range(num):
        if lst[7 * i + 1] == -1:
            continue
        obj_num += 1
        label = int(lst[7 * i + 1])
        if label == 0:
             continue
        score = lst[7 * i + 2]
        x_min = normalization(int(lst[7 * i + 3] * w), 0, w)
        y_min = normalization(int(lst[7 * i + 4] * h), 0, h)
        x_max = normalization(int(lst[7 * i + 5] * w), 0, w)
        y_max = normalization(int(lst[7 * i + 6] * h), 0, h)
        res_lst.append([score, label, x_min, y_min, x_max, y_max])
        score_lst += [score]

    # image_name, label1, score1, x_min1, y_min1, x_max1, y_max1, label2, ... ...
    return res_lst, score_lst


def get_image_lst_from_txt():
    lst = list()

    with open(txt_path, 'r') as fd:
        for line in fd.readlines():
            name = line[:line.find('.jpg')] + '.jpg'
            lst.append(os.path.join(image_dir, name))

            # lst.append(os.path.join(image_dir, line.strip('\n').split(' ')[0]))

    return lst, image_dir


def add_labels_to_image(mat, labels):
    for label in labels:
        score = str(round(label[0], 3))
        bbox = label[2:]
        cv.rectangle(mat, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv.putText(mat, score, (bbox[0], bbox[1]), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)


def mat_show(image_name, mat):
    if save_img_flag:
        cv.imwrite(save_img_path + image_name[:-4] + '.jpg', mat, [int(cv.IMWRITE_JPEG_QUALITY), 95])
    if show_img_flag:
        cv.imshow(image_name, mat)
        cv.waitKey(100)


def write_dt(image, res_lst, fd, image_dir):  # write DT into text
    # txt_path = model_path + caffemodel_name +'_dt'+'.txt'
    str_ = image.replace(image_dir + '/', '')
    str_ = str_.split("/", -1)[-2:]
    #str_ = str_[0] + "/" + str_[1]
    str_ = str_[1]
    for label in res_lst:
        for tag in label:
            str_ += ' {}'.format(str(tag))
    fd.write(str_ + '\n')

    '''
    (filepath, tempfilename) = os.path.split(image)

    print(tempfilename)

    if len(res_lst) == 0:

        dt_txt.write(tempfilename+'\n')

        print res_lst

    elif  len(res_lst) == 1:

        dt_txt.write(tempfilename + ' '+'excavator'+' ')

        print res_lst[0]

        if (res_lst[0][1]) == 1:

            dt_txt.write(str(res_lst[0][0]) + ' ' )

            anchor_box_1 = np.array(res_lst[0][2:])

            for j in anchor_box_1:

                    dt_txt.write(str(j) + ' ')

            dt_txt.write('\n')

    else:

        dt_txt.write(tempfilename + ' '+'excavator'+' ')

        print(res_lst)

        for i in res_lst:

            if i[1]==1:

                dt_txt.write('excavator' + ' ')
                #print(np.array(i[2:]))
                dt_txt.write(str(i[0]) + ' ')

                anchor_box_2 = np.array(i[2:])

                for j in anchor_box_2:

                    dt_txt.write(str(j) + ' ')

        dt_txt.write('\n')
    '''


def Calculate(net):
    params_txt = 'params.txt'
    pf = open(params_txt, 'w')
    for param_name in net.params.keys():
        weight = net.params[param_name][0].data
        # bias = net.params[param_name][1].data

        pf.write(param_name)
        pf.write('\n')

        pf.write('\n' + param_name + '_weight:\n\n')

        weight.shape = (-1, 1)

        for w in weight:
            pf.write('%ff, ' % w)
        pf.write('\n\n' + param_name + '_bias:\n\n')
        # bias.shape = (-1, 1)
        # for b in bias:
        #     pf.write('%ff, ' % b)

        pf.write('\n\n')

    pf.close


def main():
    import glob
    transformer, net = model_load()
    lst, image_dir = get_image_lst_from_txt()
    # classtransformer, classnet = classModel()

    #读取txt
    for image in lst:
        # image = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/xf_02962_xf_20200427_P001_002.jpg'
        print(image)
    #读取txt

    #读取文件夹
    # imgfile = '/media/aubopiazt/AA6CE0AF6CE07789/dataFormate/P001/0802/zt/zxc_P001_zhe_cam6_20200804162011/*.jpg'
    # for image in glob.glob(imgfile):
    #     # image = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/ROCTest/xf_02962_xf_20200427_P001_002.jpg'
    #     print(image)
    #读取文件夹

        mat = cv.imread(image, -1)
        #mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
        if mat is None:
            continue
        # mat = cv.cvtColor(mat, cv.COLOR_BGR2RGB)
        if len(mat.shape) != 3:
            print '1:', image
        elif mat.shape[2] != 3:
            print '2:', image

        folder_path, file_name = os.path.split(image)
        res_lst, score_lst = forward_img(transformer, net, mat)
        # res_lst = classForwardImg(classtransformer, classnet, mat, res_lst)
        add_labels_to_image(mat, res_lst)

        mat_show(file_name, mat)

        write_dt(image, res_lst, dt_txt, image_dir)

    dt_txt.close()


if __name__ == '__main__':
    main()

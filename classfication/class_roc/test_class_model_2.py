import os
import sys

# sys.path.append('/home/wave/caffe-segnet/python')
sys.path.append('/home/aubopiazt/linuxsoft/caffe-reid-master/python')
import caffe
import cv2
import glob
import argparse
from PIL import Image
import time
import datetime
import shutil
caffe.set_device(0)
caffe.set_mode_gpu()

modelRoot = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/datasets/caffemodel'
type = 'googleNet'
date = '20200921'
modelWeight = 'googleNet_person_2class__iter_60000.caffemodel'
modelFile = 'dev_1branch.prototxt'
#kk = os.path.join(modelRoot, type, data, modelFile)

def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--image_root_dir', type=str, default='/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/datasets/person_test/',
                        help='test image root dir')
    parser.add_argument('--save_dir', type=str, default='/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/datasets/saveResult', help='test result dir')
    parser.add_argument('--caffe_model_dir', type=str, default=os.path.join(modelRoot, type, date),
                        help='caffemodel root dir')
    parser.add_argument('--deploy', type=str,
                        default=os.path.join(modelRoot, type, modelFile),
                        help='deploy dir')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    image_root_dir = args.image_root_dir
    save_dir = args.save_dir
    caffe_model_dir = args.caffe_model_dir  # '/media/wave/HJX/seg/Models_1/test_weights.caffemodel'
    deploy = args.deploy  # '/media/wave/HJX/seg/Models_1/segnet_inference_class.prototxt'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    order = Test(image_root_dir, save_dir, caffe_model_dir, deploy)


def Test(image_root_dir, save_dir, caffe_model_dir, deploy):
    global sorce_0_all, sorce_1_all, sorce_2_all, sorce_3_all, sorce_4_all
    sorce_0_all = 0.0
    sorce_1_all = 0.0
    sorce_2_all =0.0
    #caffe_model_all = os.path.join(caffe_model_dir, "*.caffemodel")
    caffe_model_all = os.path.join(caffe_model_dir, modelWeight)
    scorce_txt_name = os.path.join(save_dir, "per_model_scorce.txt")
    per_model_scorce_file = open(scorce_txt_name, 'w')
    for caffe_model in glob.glob(caffe_model_all):
        scorce_step_name = os.path.splitext(caffe_model.split("/")[-1])[0].replace(".caffemodel", "")
        roc_name = os.path.join(save_dir,"roc.txt")
        roc_file = open(roc_name, 'w')
        net = caffe.Net(deploy, caffe_model, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        # transformer.set_mean('data', np.array([104, 117, 123]))
        transformer.set_channel_swap('data', (2, 1, 0))
        image_single_dir = os.path.join(image_root_dir, "*")
        for image_singleimage_dir in glob.glob(image_single_dir):
            image_dir = os.path.join(image_singleimage_dir, "*")
            sorce_0 = 0.0
            sorce_1 = 0.0
            sorce_2 = 0.0
            label = os.path.splitext(image_singleimage_dir.split("/")[-1])[0]
            for image in glob.glob(image_dir):
                im = caffe.io.load_image(image)
                im = cv2.resize(im, (96, 96))#150 224
                net.blobs['data'].data[...] = transformer.preprocess('data', im)
                tm1=datetime.datetime.now().microsecond
                tm3=time.mktime(datetime.datetime.now().timetuple())
                out = net.forward()
                tm2=datetime.datetime.now().microsecond
                tm4=time.mktime(datetime.datetime.now().timetuple())
                strTime='funtion time use:%dms'%((tm4-tm3)*1000.0+(tm2-tm1)/1000.0)
                print("using time: {}".format(strTime) )
                prob = net.blobs['prob'].data[0].flatten()#prediction
                # if prob[1] < 0.7 :
                #     prob[1] = 0.0
                #     prob[0] = 1.0
                # print(prob)
                order = prob.argsort()[-1]  # small-big
                print(order)
                # if order == 0 or order==2:
                #     order = 0
                if save_dir:
                    if os.path.exists(os.path.join(save_dir, "0-0")):
                        a = 1
                    else:
                        os.mkdir(os.path.join(save_dir, "0-0"))
                    if os.path.exists(os.path.join(save_dir, "0-1")):
                        a = 1
                    else:
                        os.mkdir(os.path.join(save_dir, "0-1"))
                    # if os.path.exists(os.path.join(save_dir, "0-2")):
                    #     a = 1
                    # else:
                    #     os.mkdir(os.path.join(save_dir, "0-2"))
                    if os.path.exists(os.path.join(save_dir, "1-0")):
                        a = 1
                    else:
                        os.mkdir(os.path.join(save_dir, "1-0"))
                    if os.path.exists(os.path.join(save_dir, "1-1")):
                        a = 1
                    else:
                        os.mkdir(os.path.join(save_dir, "1-1"))
                    # if os.path.exists(os.path.join(save_dir, "1-2")):
                    #     a = 1
                    # else:
                    #     os.mkdir(os.path.join(save_dir, "1-2"))
                    # if os.path.exists(os.path.join(save_dir, "2-0")):
                    #     a = 1
                    # else:
                    #     os.mkdir(os.path.join(save_dir, "2-0"))
                    # if os.path.exists(os.path.join(save_dir, "2-1")):
                    #     a = 1
                    # else:
                    #     os.mkdir(os.path.join(save_dir, "2-1"))
                    # if os.path.exists(os.path.join(save_dir, "2-2")):
                    #     a = 1
                    # else:
                    #     os.mkdir(os.path.join(save_dir, "2-2"))
                    im = Image.open(image)
                    #im.save(os.path.join(save_dir, os.path.basename(image)))
                    # im.save(os.path.join(save_dir, str(label) + "-" + str(order),
                    #                      os.path.basename(image)))
                    im.save(os.path.join(save_dir, str(label) + "-" + str(order), str(prob[order]) + '_' + os.path.basename(image)))
                    # cv2.imwrite(os.path.join(save_dir, str(label)+"-"+str(order)), im)
                roc_file.write('%s%s%s%s%s%s' % (prob[0], " ", prob[1], " ", str(label), "\n"))
                #roc_file.write('%s%s%s%s%s%s%s%s' % (prob[0], " ", prob[1], " ", prob[2], " ", str(label), "\n"))
                if int(label) == 0:
                    if int(order) == int(label):
                        sorce_0 += 1
                if int(label) == 1:
                    if int(order) == int(label):
                        sorce_1 += 1
                # if int(label) == 2:
                #     if int(order) == int(label):
                #         sorce_2 += 1
            if int(label) == 0:
                len_0 = len(glob.glob(image_dir))
                sorce_0_all = sorce_0 / len_0
            if int(label) == 1:
                len_1 = len(glob.glob(image_dir))
                sorce_1_all = sorce_1 / len_1
            # if int(label) == 2:
            #     len_2 = len(glob.glob(image_dir))
            #     sorce_2_all = sorce_2 / len_2
        roc_file.close()
        per_model_scorce_file.write('%s%s%s%s%s' % (sorce_0_all, " ", sorce_1_all, " ", "\n"))
        #per_model_scorce_file.write('%s%s%s%s%s%s%s' % (sorce_0_all, " ", sorce_1_all, " ", sorce_2_all, " ", "\n"))
    per_model_scorce_file.close()


if __name__ == '__main__':
    main()

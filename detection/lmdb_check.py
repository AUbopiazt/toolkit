# -*- coding: utf-8 -*
import lmdb
import cv2 as cv
import numpy as np
import sys
sys.path.append('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/MobileNet-YOLO-master/python')
import caffe
from caffe.proto import caffe_pb2

class checkLMDB():
    def __init__(self):
        self.dir_ = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/MobileNet-YOLO-master/data/person'
        self.label_txt = ''

    def checkLmdb(self, lmdb_name, labels, scale):
        txt_dct = {}
        '''
        with open(self.label_txt, 'r') as fd:
            for line in fd.readlines():
                lst = line.split(' ')
                name = lst[0].split('/')[-1]
                num = (len(lst) - 1) / 5
                txt_dct[name] = num
        '''
        # labels = ['other', 'stone', 'leaf', 'screw']
        env = lmdb.open(self.dir_ + '/lmdb/' + lmdb_name, readonly=True)

        lmdb_txn = env.begin()  # 生成处理句柄
        lmdb_cursor = lmdb_txn.cursor()  # 生成迭代器指针
        annotated_datum = caffe_pb2.AnnotatedDatum()  # AnnotatedDatum结构

        for key, value in lmdb_cursor:
            annotated_datum.ParseFromString(value)
            datum = annotated_datum.datum  # Datum结构
            grps = annotated_datum.annotation_group  # AnnotationGroup结构
            # type = annotated_datum.type

            image = cv.imdecode(np.fromstring(datum.data, dtype=np.uint8), -1)  # decode
            image_name = key.split('/')[-1]

            w = image.shape[1]
            h = image.shape[0]
            c = image.shape[2]
            label_num = 0
            for grp in grps:
                for box in grp.annotation:
                    xmin = box.bbox.xmin * datum.width  # Annotation结构
                    ymin = box.bbox.ymin * datum.height
                    xmax = box.bbox.xmax * datum.width
                    ymax = box.bbox.ymax * datum.height

                    xmin = box.bbox.xmin * w
                    ymin = box.bbox.ymin * h
                    xmax = box.bbox.xmax * w
                    ymax = box.bbox.ymax * h
                    cv.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
                    cv.putText(image, labels[grp.group_label], (int(xmin), int(ymin)), cv.FONT_HERSHEY_SIMPLEX, \
                               0.5, (255, 255, 0), 1)
                    label_num += 1
            if datum.channels != c:
                print
                image_name, datum.channels, c

            # str_ = 'C:{} H:{} W:{} ## [txt_N={}  lmdb_N={}]'.format(datum.channels, \
            # datum.height, datum.width, txt_dct[image_name], label_num)

            str_ = 'C:{} H:{} W:{} ## [lmdb_N={}]'.format(datum.channels, \
                                                          datum.height, datum.width, label_num)
            cv.putText(image, str_, (5, 15), cv.FONT_HERSHEY_SIMPLEX, \
                       0.5, (0, 0, 255), 1)
            image = cv.resize(image, scale, interpolation=cv.INTER_CUBIC)

            cv.imshow(image_name, image)  # 显示图片
            cv.moveWindow(image_name, 0, 0)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    checklmdb = checkLMDB()
    checklmdb.checkLmdb('person_trainval_lmdb', labels=['groupback', 'excavator', 'truck', 'wheel'], scale=(1920, 1080))
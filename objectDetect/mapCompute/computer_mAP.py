from voc_eval import voc_eval

# print voc_eval('/home/lzc/wendang/darknet/results/{}.txt', '/home/lzc/wendang/darknet/VOC2018/VOCdevkit/VOC2018/Annotations/{}.xml', '/home/lzc/wendang/darknet/VOC2018/VOCdevkit/VOC2018/ImageSets/Main/all.txt', 'person', '.')

rec, prec, ap = voc_eval('/home/lzc/wendang/darknet/results/{}.txt',
                         '/home/lzc/wendang/darknet/VOC2018/VOCdevkit/VOC2018/Annotations/{}.xml',
                         '/home/lzc/wendang/darknet/VOC2018/VOCdevkit/VOC2018/ImageSets/Main/all.txt', 'person', '.')

print('rec', rec)
print('prec', prec)
print('ap', ap)
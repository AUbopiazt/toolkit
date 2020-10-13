from voc_eval import voc_eval
 
import os
 
current_path = os.getcwd()
#results_path = current_path+"results"
sub_files = os.listdir(current_path)
 
mAP = []
for i in range(len(sub_files)):
    theDir=sub_files[i]
    txtDir='txt'
    print theDir
    if theDir[-3:]==txtDir:
        class_name = sub_files[i].split(".txt")[0]
        rec, prec, ap = voc_eval('/home/lzc/wendang/darknet/results/{}.txt', '/home/lzc/wendang/darknet/VOC2018/VOCdevkit/VOC2018/Annotations/{}.xml', '/home/lzc/wendang/darknet/VOC2018/VOCdevkit/VOC2018/ImageSets/Main/all.txt', class_name, '.')
        print("{} :\t {} ".format(class_name, ap))
        mAP.append(ap)
 
mAP = tuple(mAP)
 
print("***************************")
print("mAP :\t {}".format( float( sum(mAP)/len(mAP)) )) 

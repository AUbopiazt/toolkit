# -*- coding:utf-8 -*-
import os
import cv2
import glob
import numpy as np
import shutil
import json


rootpath = '/media/aubopiazt/reid/zt/GigaVision_modify/'
savepath = '/media/aubopiazt/reid/zt/fupai/'

for root, dirs, file in os.walk(rootpath):
    for dir in dirs:
        imgs = rootpath + dir + '/*'
        for img in glob.glob(imgs):
            shutil.move(img, savepath)



# img_dir = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/result_img/tiny_1/*.jpg'
# video_dir = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/video1.avi'
# fps = 25
# img_size = (1080, 1920)
#
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# videoWriter = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, img_size)
#
# for img in glob.glob(img_dir):
#     print(img)
#     imgs = cv2.imread(img)
#     videoWriter.write(imgs)
# videoWriter.release()




# videoPath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/data/2020-09-15/194.168.1.62_01_20200915094744340.mp4'
# savepath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/data/2020-09-15/video0/'
# cap = cv2.VideoCapture(videoPath)
# count = 0
# while 1:
#     ret, frame = cap.read()
#     #cv2.imshow('cap', frame)
#     #if cv2.waitKey(100) & 0xff == ord('q'):
#     if not ret:
#         break
#     count = count + 1
#     if count < 500 or count % 5 != 0:
#         continue
#     newname = savepath + str(count-500).rjust(6, '0') + '.jpg'
#     cv2.imwrite(newname, frame)
#     print(count)
# cap.release()
# cv2.destroyAllWindows()



# count = 0
# imgpath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3/caffe-yolov3-master/images/person/'
# savepath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3/caffe-yolov3-master/images/person/'
# pathlists = os.listdir(imgpath)
# for pathlist in pathlists:
#     #newname = pathlist[:-4].rjust(4, '0')+'.jpg'
#     count = count + 1
#     newname = str(count).rjust(4, '0') + '.jpg'
#     shutil.move(imgpath+pathlist, savepath+newname)



# def writerJsonData(saveJsonPath, dict):
#     with open(saveJsonPath, 'w') as w:
#         json.dump(dict, w, indent=4)
#     w.close()
#
# jsonpath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/persontrain/person_masic/origin_10/*.json'
# newpath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/persontrain/person_masic/origin_5_1/'
# count = 0
# for jsonfile in glob.glob(jsonpath):
#     count = count + 1
#     print(count)
#     imgfile = jsonfile.replace('json', 'jpg')
#     jsonBaseName = os.path.basename(jsonfile)
#     newJsonName = jsonBaseName[:-5] + '_masic_10.json'
#     newImagename = jsonBaseName[:-5] + '_masic_10.jpg'
#
#
#     # imgfile = jsonfile.replace('json', 'jpg')
#     # imgname = imgfile.split('/')[-1]
#     # jsonname = imgname.replace('jpg', 'json')
#     if os.path.exists(imgfile):
#         f = open(jsonfile, 'r', encoding='gb2312')
#         param = json.load(f)
#         param['imagePath'] = newImagename
#         param['imageData'] = None
#         newjson = newpath + newJsonName
#         newimg = newpath + newImagename
#         writerJsonData(newjson, param)
#         shutil.move(imgfile, newimg)








# count = 0
# imgpath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3/caffe-yolov3-master/images/person/'
# savepath = '/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3/caffe-yolov3-master/images/person/'
# pathlists = os.listdir(imgpath)
# for pathlist in pathlists:
#     #newname = pathlist[:-4].rjust(4, '0')+'.jpg'
#     count = count + 1
#     newname = str(count).rjust(4, '0') + '.jpg'
#     shutil.move(imgpath+pathlist, savepath+newname)





# imgpath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/part15'
# savepath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin'
# pathlists = os.listdir(imgpath)
# for pathlist in pathlists:
#     imglists = os.listdir(imgpath + '/' + pathlist)
#     for imglist in imglists:
#         ront = imgpath + '/' + pathlist + '/' + imglist + '/*.jpg'
#         for img in glob.glob(ront):
#             jsons = img.replace('jpg', 'json')
#             shutil.move(img, savepath)
#             shutil.move(jsons, savepath)



# def writerJsonData(saveJsonPath, dict):
#     with open(saveJsonPath, 'w') as w:
#         json.dump(dict, w, indent=4)
#     w.close()
#
# jsonpath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/persontrain/test/*.json'
# newpath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/persontrain/test/'
# count = 0
# for jsonfile in glob.glob(jsonpath):
#     count = count + 1
#     print(count)
#     imgfile = jsonfile.replace('json', 'jpg')
#     imgname = imgfile.split('/')[-1]
#     jsonname = imgname.replace('jpg', 'json')
#     if os.path.exists(imgfile):
#         f = open(jsonfile, 'r', encoding='gb2312')
#         param = json.load(f)
#         param['imagePath'] = imgname
#         param['imageData'] = None
#         newjson = newpath + jsonname
#         newimg = newpath + imgname
#         writerJsonData(newjson, param)
#         shutil.copy(imgfile, newimg)




# def writerJsonData(saveJsonPath, dict):
#     with open(saveJsonPath, 'w') as w:
#         json.dump(dict, w)
#     w.close()
# path = 'F:\\zt\\excavator\\excavator_part1\\*.json'
# for jsonfile in glob.glob(path):
#     newjsonfile = jsonfile.replace('excavator_part1', 'new_excavator')
#     imgfile = jsonfile.replace('json', 'jpg')
#     newimgfile = imgfile.replace('excavator_part1', 'new_excavator')
#     ff = open(jsonfile, 'r', encoding='gb2312')
#     jsontxt = json.load(ff)
#
#     shape = jsontxt['shapes']








# saveflag = 0
# path = 'D:\\周报\\zt\\P001\\part13\\zxc_done\\*.jpg'
# savepath = 'D:\\周报\\zt\\P001\\part13\\reid_seg_check\\'
# count = 0
# for imgfile in glob.glob(path):
#     jsonfile = imgfile.replace('jpg', 'json')
#     if os.path.exists(jsonfile):
#         shutil.move(jsonfile, savepath)
#         shutil.move(imgfile, savepath)
#         count = count + 1
#         print(count)






# truck = ['truck']
# wheel = ['wheel']
# kk = 0
# for jsonfile in glob.glob(path):
#     saveflag = 0
#     imgfile = jsonfile.replace('json', 'jpg')
#     ff = open(jsonfile, 'r', encoding='gb2312')
#     jsontext = json.load(ff)
#     shapes = jsontext['shapes']
#     for shape in shapes:
#         label = shape['label']
#         if label in truck or label in wheel:
#             saveflag = 1
#             break
#     ff.close()
#     if saveflag == 1:
#         shutil.move(imgfile, imgfile.replace('trainDatas', 'truck_wheel'))
#         shutil.move(jsonfile, jsonfile.replace('trainDatas', 'truck_wheel'))







# path = 'F:\\zt\\excavator\\excavator_part1\\*.json'
# #savePath = 'E:\\zt\\trainsData_deleteTruck\\trainsData_deletCarAndTruck\\'
# def writerJsonData(saveJsonPath, dict):
#     with open(saveJsonPath, 'w') as w:
#         json.dump(dict, w, indent=4)
#     w.close()
#
# def getJsonData(path1):
#     num = 0
#     for jsonPath in glob.glob(path1):
#         num = num + 1
#         print(num)
#         savejson = jsonPath.replace('excavator_part1', 'new_excavator')
#         #saveJsonName = jsonPath.split('\\')[-1]
#         #savejson = savePath + saveJsonName
#         oldImage = jsonPath.replace('json', 'jpg')
#         newImage = oldImage.replace('excavator_part1', 'new_excavator')
#         dict = {}
#         f = open(jsonPath, 'r', encoding='gb2312')
#         params = json.load(f)
#         labelNum = len(params['shapes'])
#         deletNum = 0
#         count = 0
#         for i in range(len(params['shapes'])):
#             assert params['shapes'][count]['label'] in ['car', 'truck', 'truckhead', 'excavator', 'pile_driver', 'push_bench', 'execavator', 'other', 'wheel'], params['shapes'][count]['label']
#             if params['shapes'][count]['label'] in ['car', 'truck', 'truckhead', 'other', 'wheel']:
#                 del params['shapes'][count]
#                 count = count - 1
#                 deletNum = deletNum + 1
#             count = count + 1
#         f.close()
#         if labelNum > deletNum:
#             writerJsonData(savejson, params)
#             shutil.copy(oldImage, newImage)
#
# #for jsonfile in glob.glob(path):
# getJsonData(path)





# frompath = 'D:\\周报\\zt\\p003\\第一批标定\\zt1\\*.json'
# for jsonfile in glob.glob(frompath):
#     imgfile = jsonfile.replace('json', 'jpg')
#
#     newjsonfile = jsonfile.replace('zt1', 'zt_done')
#     newimgfile = imgfile.replace('zt1', 'zt_done')
#     shutil.copy(jsonfile, newjsonfile)
#     shutil.copy(imgfile, newimgfile)


# path = 'D:\\周报\\zt\\part8\\zt\\sgl\\sgl\\*.jpg'
#
# for img in glob.glob(path):
#     imgName = 'sgl_' + img.split('\\')[-1]
#     newpath = 'D:\\周报\\zt\\part8\\zt\\sgl2\\' + imgName
#     shutil.copy(img, newpath)
#
#
# savepath = 'D:\\周报\\zt\\part7\\调整\\truckOrExcavator\\'
# TruckOrExcavator = ['truck', 'excavator']
# count = 0
# for jsonFile in glob.glob(path):
#     flag = 0
#     with open(jsonFile, 'r', encoding='utf8') as f:
#         jsonData = json.load(f)
#         labels = jsonData['shapes']
#         for i in range(len(labels)):
#             if labels[i]['label'] in TruckOrExcavator:
#                 flag = 1
#                 break
#     f.close()
#     if flag == 1:
#         shutil.move(jsonFile, savepath)
#         shutil.move(jsonFile.replace('json', 'jpg'), savepath)
#         count = count + 1
#         print('count:', count)
#         flag = 0
#





# paths = 'E:\\BaiduNetdiskDownload\\OTS_ALPHA\\haze\\OTS\\*.jpg'
# labelPaht = 'E:\\BaiduNetdiskDownload\\OTS_ALPHA\\clear\\haze_0.9_0.12'
# for imgName in glob.glob(paths):
#     if imgName.find('_0.95_0.2.jpg') != -1:
#         shutil.copy(imgName, labelPaht)
# gt = 'D:\\BaiduNetdiskDownload\\CVPR19RainTrain\\train\\gt\\'
# I = 'D:\\BaiduNetdiskDownload\\CVPR19RainTrain\\train\\in\\*'
# S = 'D:\\BaiduNetdiskDownload\\CVPR19RainTrain\\train\\streak\\'
# T = 'D:\\BaiduNetdiskDownload\\CVPR19RainTrain\\train\\trans\\*'
# A = 'D:\\BaiduNetdiskDownload\\CVPR19RainTrain\\train\\atm\\'
# savepath = 'D:\\BaiduNetdiskDownload\\CVPR19RainTrain\\train\\haze\\'
# count = 0
# for transissionPath in glob.glob(T):
#     name = transissionPath.split('\\')[-1].split('_s')[0] + '.png'
#     gtPath = gt + name
#     transiImage = cv2.imread(transissionPath)/255.0
#     AtmImage = cv2.imread(transissionPath.replace('trans', 'atm'))/255.0
#     gtImage = cv2.imread(gtPath)/255.0
#     haze = np.empty(gtImage.shape, gtImage.dtype)
#     for ind in range(3):
#         haze[:, :, ind] = gtImage[:, :, ind] * transiImage[:, :, ind] + AtmImage[:, :, ind] * (1 - transiImage[:, :, ind])
#     haze = np.clip(haze, 0, 1)
#     cv2.imwrite(transissionPath.replace('trans', 'haze'), haze * 255)


    # count = count + 1
    # name = haze.split('\\')[-1]
    # streak = S + name
    # transission = T + name
    # Atm = A + name
    # SS = cv2.imread(haze, -1) / 255.0
    # streakImage = cv2.imread(streak) / 255.0
    # tranImage = cv2.imread(transission) / 255.0
    # atmImage = cv2.imread(Atm) / 255.0
    # saveimage = np.empty(SS.shape, SS.dtype)
    # for ind in range(0, 3):
    #     #saveimage[:, :, ind] = (SS[:, :, ind] - atmImage[0, 0, 0] * (1 - tranImage[:, :, ind]))/tranImage[:, :, ind]
    # cv2.imwrite(savepath + str(count) + '.png', saveimag
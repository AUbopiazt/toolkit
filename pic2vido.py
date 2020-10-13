# -*- coding:utf-8 -*-
import cv2
from PIL import Image
import glob
import os
def  jpg_to_video(path, fps):
    """ 将图片合成视频. path: 视频路径，fps: 帧率 """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    images = os.listdir('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/result_img/tiny_2')#os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    image = Image.open('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/result_img/tiny_2/' + images[0])
    vw = cv2.VideoWriter(path, fourcc, fps, image.size)

    #os.chdir('result')
    #for i in range(len(images)):
    for jpgfile in glob.glob('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet/result_img/tiny_2/*.jpg'):
        # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
        #jpgfile = '/media/wave/1CCB-80A2/dark_video/result/8192kbps/'+'%05d' % int(i+1) + '.png'
        try:
            new_frame = cv2.imread(jpgfile)
            vw.write(new_frame)
        except Exception as exc:
            print(jpgfile, exc)
    vw.release()
    print(path, 'Synthetic success!')


if __name__ == '__main__':
  PATH_TO_OUTCOME = os.path.join('/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/ObjectDetect/caffe-yolov3_4/darknet', 'video2.mp4')
  jpg_to_video(PATH_TO_OUTCOME, 25)  # 图片转视频

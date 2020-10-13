from albumentations import (HorizontalFlip, IAAPerspective, ShiftScaleRotate,
                            CLAHE, RandomRotate90, Transpose, Blur, OpticalDistortion,
                            GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise,
                            GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
                            IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose)
import numpy as np
import cv2
import glob
import os

#image = cv2.imread('/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/ropes/ropes_self/cwf_zt_20200630I_P001_SafetyRope_001_20_661_511_347_567.jpg')
rootPath = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/person/'
def strong_aug(p=1.0):
    return Compose([#RandomRotate90(),
                  Flip(),  #翻转
                  #转置
                  #Transpose(),
                  #高斯噪点
                  OneOf([IAAAdditiveGaussianNoise(),
                         GaussNoise(),
                         ], p=0.2),
                  #模糊
                  OneOf([MotionBlur(p=0.2),
                         MedianBlur(blur_limit=3, p=0.1),
                         ], p=0.2),
                  ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75),  # 随机仿射变换
                  #畸变
                  OneOf([OpticalDistortion(p=0.3),
                         GridDistortion(p=0.1),
                         IAAPiecewiseAffine(p=0.3),
                         ], p=0.2),
                  #锐化
                  OneOf([CLAHE(clip_limit=2),
                         IAASharpen(),
                         IAAEmboss(),
                         RandomBrightnessContrast(),
                         ], p=0.3),
                  HueSaturationValue(p=0.5),
                  ], p=p)
if __name__ == '__main__':

    for root, dirs, files in os.walk(rootPath):
        for dir in dirs:
            savepath = root + dir + '_augmentation'
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            for images in glob.glob(os.path.join(root, dir) + '/*'):
                augumentation = strong_aug(p=1.0)
                image = cv2.imread(images)
                auguImage = augumentation(image=image)['image']
                imageSavePath = savepath + '/' + images.split('/')[-1]
                cv2.imwrite(imageSavePath, auguImage)
#
# image1 = Flip(p=1)(image=image)['image']
# image2 = ShiftScaleRotate(p=1)(image=image)['image']
# image3 = Compose([CLAHE(),#对比度受限直方图均匀
#                   RandomRotate90(),#随机旋转90
#                   Transpose(),#转置
#                   ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75),#随机仿射变换
#                   Blur(blur_limit=3),#模糊
#                   OpticalDistortion(),#光学畸变
#                   GridDistortion(),#网格畸变
#                   HueSaturationValue()#随机改变HUE，饱和度和值
#                   ], p=1.0)(image=image)['image']
# image4 = Compose([RandomRotate90(),
#                   Flip(),  #翻转
#                   #转置
#                   Transpose(),
#                   #高斯噪点
#                   OneOf([IAAAdditiveGaussianNoise(),
#                          GaussNoise(),
#                          ], p=0.2),
#                   #模糊
#                   OneOf([MotionBlur(p=0.2),
#                          MedianBlur(blur_limit=3, p=0.1),
#                          ], p=0.2),
#                   ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75),  # 随机仿射变换
#                   #畸变
#                   OneOf([OpticalDistortion(p=0.3),
#                          GridDistortion(p=0.1),
#                          IAAPiecewiseAffine(p=0.3),
#                          ], p=0.2),
#                   #锐化
#                   OneOf([CLAHE(clip_limit=2),
#                          IAASharpen(),
#                          IAAEmboss(),
#                          RandomBrightnessContrast(),
#                          ], p=0.3),
#                   HueSaturationValue(p=0.3),
#                   ], p=1.0)(image=image)['image']
# cv2.imwrite('./1.jpg',image1)
# cv2.imwrite('./2.jpg',image2)
# cv2.imwrite('./3.jpg', image3)
# cv2.imwrite('./4.jpg', image4)

# coding: utf-8

# 微信公众号：深度学习与神经网络  
# Github：https://github.com/Qinbf  
# 优酷频道：http://i.youku.com/sdxxqbf  

# In[5]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

caffe_root = '/media/wavereid/DATA/zt/objectDetect/mobilenet-yolo/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
import glob
import time


# In[7]:

#网络结构描述文件
#deploy_file = '/media/wavereid/DATA/zt/objectDetect/mobilenet-yolo/models/MobileNet/snapshot/zt_model/coco/mobilenet_deploy_class3_coco.prototxt'#"/media/wavereid/DATA/zt/toolkit/no_bn.prototxt"
# #模型文件
#model_file  = '/media/wavereid/DATA/zt/objectDetect/mobilenet-yolo/models/MobileNet/snapshot/zt_model/coco/mobilenet_yolov3_coco_320_excavotor__iter_34000.caffemodel'#"/media/wavereid/DATA/zt/toolkit/no_bn.caffemodel"
#网络结构描述文件
deploy_file = "/media/wavereid/DATA/zt/toolkit/no_bn.prototxt"
# #模型文件
model_file  = "/media/wavereid/DATA/zt/toolkit/no_bn.caffemodel"
#测试图片
test_data   = "/media/wavereid/DATA/zt/toolkit/pic/hyt_0.jpg"
#特征图路径
feature_map_path = "/media/wavereid/DATA/zt/toolkit/"
pathPic = '/media/wavereid/DATA/zt/toolkit/pic/hyt_0.jpg'

#编写一个函数，用于显示各层的参数,padsize用于设置图片间隔空隙,padval用于调整亮度 
def show_data(data, name, padsize=1, padval=0):
    
    #归一化
    data -= data.min()
    data /= data.max()
    
    #根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))
    
    # 对于conv1，data.shape->(20,24,24)
    # （前面填补0个，后面填补n ** 2 - data.shape[0]），（前面填补0个，后面填补padsize个），（前面填补0个，后面填补padsize个）
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize))
    data = np.pad(data, padding, mode='constant', constant_values=padval)#常数值填充，填充0
    
    # 对于conv1，padding后data.shape->(25,25,25)
    # 对于conv1，将(25,25,25)reshape->(5,5,25,25)再transpose->(5,25,5,25)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3))
    
    # 再将（n, W, n, H）变换成(n*w, n*H)
    data = data.reshape((n * data.shape[1], n * data.shape[3]))
    

    image_path = os.path.join(feature_map_path,name)#特征图路径
    plt.set_cmap('gray')#设置为灰度图
    plt.imsave(image_path,data)#保存生成的图片
    plt.axis('off')#不显示坐标
    
    print name
    #显示图片
    img=Image.open(image_path)
    plt.imshow(img)
    plt.show()

def loadModel(protoPath, caffemodelPath):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(protoPath, caffemodelPath, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    mu = np.array([1, 1, 1])
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 0.007843)
    return net, transformer

def featureVisual(net, transformer, img, layer_name, save_name):
    '''
    caffe input BGR(0---255),
    if use cv2.imread for img, satisfy the caffe
    elseif use use caffe.io.load_image or PIL
    python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
    transformer.set_raw_scale('data', 255)      # 缩放到[0，255]之间
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR（caffe中图片是BGR格式，而原始格式是RGB，所以要转化）
    '''
    #img = caffe.io.load_image(test_data, color=True)
    #img = cv2.imread(test_data)
    transformer = transformer.preprocess('data', img)
    net.blobs['data'].data[...] = transformer
    net.forward()
    feature = net.blobs[layer_name].data  # b c h w
    show_data(feature[0], save_name)

if __name__ == '__main__':

    '''
    caffe input BGR(0---255),
    if use cv2.imread for img, satisfy the caffe
    elseif use use caffe.io.load_image or PIL
    python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
    transformer.set_raw_scale('data', 255)      # 缩放到[0，255]之间
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR（caffe中图片是BGR格式，而原始格式是RGB，所以要转化）
    '''
    net, transformer = loadModel(deploy_file, model_file)
    print [(K, V[0].data.shape) for K, V in net.params.items()]
    #img = caffe.io.load_image(test_data, color=True)
    img = cv2.imread(test_data)
    featureVisual(net, transformer, img, 'conv1/dw', 'conv1.jpg')



#----------------------------数据预处理---------------------------------
'''
#初始化caffe
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(deploy_file, #网络结构描述文件 
                model_file,  #训练好的模型
                caffe.TEST)  #使用测试模式

#输出网络每一层的参数
print [(k, v[0].data.shape) for k, v in net.params.items()]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# python读取的图片文件格式为H×W×K(高度，宽度，通道数)，需转化为K×H×W（通道数，高度，宽度）
transformer.set_transpose('data', (2, 0, 1))

# python中将图片存储为[0-1]
# 如果模型输入用的是0~255的原始格式，则需要做以下转换
# transformer.set_raw_scale('data', 255)

# caffe中图片是BGR格式，而原始格式是RGB，所以要转化
#transformer.set_channel_swap('data', (2, 1, 0))
mu = np.array([1, 1, 1])  #
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 0.007843)

#----------------------------数据运算---------------------------------
#读取图片
#参数color: True(default)是彩色图，False是灰度图
#img = caffe.io.load_image(test_data,color=True)


img = cv2.imread(test_data)

# 数据输入、预处理
net.blobs['data'].data[...] = transformer.preprocess('data', img)
kk = net.blobs['data'].data.shape
# 将输入图片格式转化为合适格式（与deploy文件相同）
#net.blobs['data'].reshape(1, 3, 320, 512)

# 前向迭代，即分类。保存输出
out = net.forward()
# 输出结果为各个可能分类的概率分布
#print "Prob:"
print out['detection_out']

#最可能分类
predict = out['detection_out'].argmax()
print "Result:" + str(predict)

#----------------------------输出特征图---------------------------------
#第一个卷积层输出的特征图
feature = net.blobs['conv0'].data
show_data(feature[0],'conv1.jpg')
#第一个池化层输出的特征图
feature = net.blobs['pool1'].data
show_data(feature.reshape(20,12,12),'pool1.jpg')
#第二个卷积层输出的特征图
feature = net.blobs['conv2'].data
show_data(feature.reshape(50,8,8),'conv2.jpg')
#第二个池化层输出的特征图
feature = net.blobs['pool2'].data
show_data(feature.reshape(50,4,4),'pool2.jpg')
'''




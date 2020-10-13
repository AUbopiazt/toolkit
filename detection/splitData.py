import os  
import random 
xmlfilepath=r'/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/VOCdevkit/excavator/Annotations/'   #change xml path
saveBasePath=r"/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/VOCdevkit/excavator/ImageSets"     #change base path
trainval_percent=0.85                                             #adjust trainval percentage
train_percent=0.9                                                #adjust train percentage
total_xml = os.listdir(xmlfilepath)
num=len(total_xml)    
list=range(num)    
tv=int(num*trainval_percent)    
tr=int(tv*train_percent)    
trainval= random.sample(list,tv)    
train=random.sample(trainval,tr)    
  
print("train and val size",tv)  
print("traub suze",tr)  
ftrainval = open(os.path.join(saveBasePath,'Main/trainval.txt'), 'w')    
ftest = open(os.path.join(saveBasePath,'Main/test.txt'), 'w')    
ftrain = open(os.path.join(saveBasePath,'Main/train.txt'), 'w')    
fval = open(os.path.join(saveBasePath,'Main/val.txt'), 'w')    
  
for i  in list:    
    name=total_xml[i][:-4]+'\n'    
    if i in trainval:    
        ftrainval.write(name)    
        if i in train:    
            ftrain.write(name)    
        else:    
            fval.write(name)    
    else:    
        ftest.write(name)    
    
ftrainval.close()    
ftrain.close()    
fval.close()    
ftest .close()  

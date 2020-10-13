import os
import shutil
import glob
import cv2

# imgpaths = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/excavator/*.jpg'
# jsonpaths = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/excavator/'
# origin = '/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/detect/excavator_wheel/origin/'
#
# for img in glob.glob(imgpaths):
#     jsonname = os.path.basename(img).replace('jpg', 'json')
#     jsonpath = origin+jsonname
#     shutil.copy(jsonpath, jsonpaths)


imgpaths = '/media/aubopiazt/reid/zt/GigaVision/'
saveroot = '/media/aubopiazt/reid/zt/GigaVision_modify/'
items = '04_Primary_School'

imgs = imgpaths + items + '/*.jpg'
savepath = saveroot + items
if os.path.exists(savepath):
    shutil.rmtree(savepath)
os.mkdir(savepath)
for img in glob.glob(imgs):
    print('img:', img)
    basename = os.path.basename(img)
    pic = cv2.imread(img)
    height, width, C = pic.shape
    dim = (width/10, height/10)
    resized = cv2.resize(pic, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(savepath+'/'+basename, resized)




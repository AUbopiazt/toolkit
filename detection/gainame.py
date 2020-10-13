import os
import glob
import cv2
import shutil

eoot = '/media/wave/1CCB-80A2/SR/video_test_hr_1080/' + "*.png"

for i in glob.glob(eoot):
    print(i)
    name = os.path.basename(i).replace('.png','')
    #name.zfill(4)
    name = '%05d' % int(name)
    shutil.copyfile(i, '/media/wave/1CCB-80A2/SR/video_test_hr_1080/'+name+'.png')
    #imag = cv2.imread(i)
    #cv2.imwrite('/media/wave/1CCB-80A2/SR/xinmoxing_2x/'+name+'.png',imag)

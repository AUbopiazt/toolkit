#coding=utf-8
import glob
import shutil

path = '/media/aubopiazt/大白菜U盘/zt/2020-08-05/safety_rope/*'
savepath = '/media/aubopiazt/大白菜U盘/zt/2020-08-05/safety_rope/'
i = 0
for video in glob.glob(path):
    newname = 'zt_20200805O_P001_SafetyRope_00' + str(i) + '.mp4'
    i = i + 1
    shutil.move(video, savepath + newname)

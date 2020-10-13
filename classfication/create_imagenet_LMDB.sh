
phase=val #train ro test

DATA=/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/person/

IMGLIST=/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/person/${phase}.lst
LMDBNAME=/media/aubopiazt/AA6CE0AF6CE07789/ubuntufile/datasets/classfication/person/${phase}_LMDB

rm -r $LMDBNAME
echo 'converting images...'
/home/aubopiazt/linuxsoft/caffe-reid-master/build/tools/convert_imageset --shuffle=true --resize_width=96 --resize_height=96 $DATA $IMGLIST $LMDBNAME/

# -*- coding:utf-8
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
classNumber = 2
def get_output_file1(output_file):
    prod_all=[]
    label_all=[]
    for line in open(output_file):
        x = line.split()
        prod=[]
        label=[]
        for i in range(int(classNumber)):
                prod.append(float(x[i]))
        tag = int(x[classNumber])#int(x[classNumber])
        for j in range(classNumber):
            if (j == tag):
                label.append(1)
            else:
                label.append(0)
        prod_all.append(prod)
        label_all.append(label)
    return  prod_all,label_all
def ROC(prod_all,label_all,classLabel,rgb="r",leged="line"):
    y_true = np.array(label_all)
    y_predict = np.array(prod_all)
    fpr, tpr, thr = roc_curve(y_true[:, classLabel], y_predict[:, classLabel])
    fid = open( "/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/datasets/saveResult/roc_6w_th.txt", 'a+')#( "E:\\TestProject\\DrivingTest\\20190423_S_U_V_T\\TestResult\\U\\q_4_fpr_tpr_thr_del_squeesev1.1__iter_4000.txt", 'a+')
    fid.writelines(str(classLabel)+"\n"+" fpr  tpr   thr"+"\n")
    for i in range(len(fpr)):
        fid.writelines( str(fpr[i])+" "+str(tpr[i])+" "+str(thr[i])+"\n")
    plt.plot(fpr, tpr,  clip_on=False,color=rgb,label=leged)
    AUC=auc(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='best')
    return AUC
if __name__ == '__main__':
    output_file1 = "/media/aubopiazt/BA6ED4596ED40FCD/ubuntufile/datasets/saveResult/roc.txt"#"E:\\TestProject\\DrivingTest\\20190423_S_U_V_T\\TestResult\\U\\del_squeesev1.1__iter_4000.txt"
    prod_all, label_all = get_output_file1(output_file1)
    AUC0 = ROC(prod_all, label_all, 0, rgb="r", leged="other")#other_0
    AUC1 = ROC(prod_all, label_all, 1, rgb="g", leged="person")#gun_1
    #AUC2 = ROC(prod_all, label_all, 2, rgb="b", leged="rope")#knife_2
    # AUC3 = ROC(prod_all, label_all, 3, rgb="purple", leged="gun_3")#phone_3
    # AUC4 = ROC(prod_all, label_all, 4, rgb="yellow", leged="stick_4")
    #plt.title("AUC(0 = %.4f,1 = %.4f,2 = %.4f,3 = %.4f,4 = %.4f)" % (AUC0, AUC1,AUC2,AUC3,AUC4))
    #plt.title("AUC(0 = %.4f,1 = %.4f,2 = %.4f,3 = %.4f)" % (AUC0, AUC1, AUC2, AUC3))
    plt.title("AUC(other = %.4f,person = %.4f)" % (AUC0, AUC1))
    plt.show()
    #plt.savefig("/media/wave/2838F81538F7E02C/zuixin/model_3_result_class/model_3_iter_72500roc.jpg")


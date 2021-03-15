import os
import cv2
import numpy as np


def main():
    path='/save/2017018/rconda01/KITTI_dataset_3/training/images/'
    suffix = '_Vl.png'

    totals = []
    nbpixels = []
    for i in range(len(os.listdir(path))):
        if i%600==0:
            print(i)
        folder = os.listdir(path)[i]
        img = cv2.imread(path+folder+'/'+folder+suffix,0)/255.0
        nbpixels.append(img.size)
        totals.append(img.sum())

    mean = sum(totals)/sum(nbpixels)
    print("Mean : {}".format(sum(totals)/sum(nbpixels)))

    totals_2 = []
    for i in range(len(os.listdir(path))):
        if i%600==0:
            print(i)
        folder = os.listdir(path)[i]
        img = cv2.imread(path+folder+'/'+folder+suffix,0)/255.0
        img=np.power(img-mean,2)
        totals_2.append(img.sum())

    print("Std : {}".format(np.sqrt(sum(totals_2)/(sum(nbpixels)-1))))
    
if __name__ == '__main__':
    main()

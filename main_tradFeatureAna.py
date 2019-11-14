# -*- coding: UTF-8 -*-
import cv2
import os
import math
import numpy as np
from tifffile import TiffFile
import matplotlib.pyplot as plt


def feature_ana(tiff_path, save_path):
    plt.figure(figsize=(20,14))
    # tif = TiffFile(tiff_path)
    image_arr = cv2.imread(tiff_path, 0) # tif.asarray()
    rows = 4
    cols = math.ceil((image_arr.shape[0] + 1)*1.0/rows)
    print("rows:{}-cols:{}-frame{}".format(rows, cols, image_arr.shape[0]))
    dys = []
    for i in range(image_arr.shape[0]):
        arr = cv2.medianBlur(image_arr[i, :, :], ksize=1)
        dys.append(cv2.Sobel(arr,cv2.CV_32F,0, 1, ksize=1))
    plt.subplot(rows, cols, 1)
    plt.imshow(image_arr[1].T[:600, :], cmap="gray")
    maxidx = 0
    sum = 0.0
    for i in range(1, image_arr.shape[0] + 1):
        if i == 1:
            image = np.expand_dims(image_arr[i-1], axis=2)
            image = np.concatenate((image, image, image), axis=-1)
            height = image.shape[0] - 1
            shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
            gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            index = [0 if len(np.where(thresh[:, i] > 0)[0]) == 0 else height - np.where(thresh[:, i] > 0)[0][0] for i in
                     range(thresh.shape[1])]
            maxidx = np.max(index)
            print("max of index is {}".format(maxidx))
        mean = np.mean(image_arr[i-1][image_arr[i-1].shape[0]-maxidx:])
        std = float(np.std(image_arr[i-1][image_arr[i-1].shape[0]-maxidx:]))
        sum +=mean
        plt.subplot(rows,cols,i+1)
        plt.plot(np.gradient(np.mean(image_arr[i-1], 1)))
        plt.plot(np.mean(image_arr[i-1], 1))
        plt.title("avg: {}--std: {}".format(int(mean), round(std, 2)))
        # plt.legend("avg-intensity: {}".format(mean))
    print("mean intensity of all frame is {}".format(sum/image_arr.shape[0]))
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    save_dir = "./images/"
    travel_dirs = ['normal', 'hsil', 'cancer']
    prefix = ['normal', 'HSIL', 'cancer']
    for idx, td in enumerate(travel_dirs):
        inner_dir = os.path.join(save_dir, travel_dirs[idx])
        tiff_paths = [tp for tp in os.listdir(inner_dir) if tp.endswith('.png')]
        for tp in tiff_paths:
            print(tp)
            save_name = prefix[idx] + "-" + tp[:-5] + '.png'
            feature_ana(os.path.join(inner_dir, tp), os.path.join(save_dir, save_name))
            break

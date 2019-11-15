# -*- coding: UTF-8 -*-
import cv2
import os
import math
import numpy as np
# from tifffile import TiffFile
import matplotlib.pyplot as plt
from scipy import optimize


def fmax(x, a, b, c, duration):
    return a*np.sin(x*np.pi/duration+b)+c


def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D


def f_4(x, A, B, C, D, E):
    return A*x*x*x*x + B*x*x*x + C*x*x + D*x + E;


def feature_ana(tiff_path, save_path, delta = 1):
    plt.figure(figsize=(20,14))
    # tif = TiffFile(tiff_path)
    image = cv2.imread(tiff_path, 0)[200:, :]
    image = cv2.medianBlur(image, 7)
    image_arr = image[:, :30]
    print('shape:{}'.format(image_arr.shape))
    row, column = math.ceil(image_arr.shape[0]/delta), image_arr.shape[1]
    image_arr = image_arr.astype("float")

    gradient = np.zeros((row, column))

    x = np.arange(0, image_arr[:, 0].shape[0], 1)
    popt, pcov = optimize.curve_fit(f_4, x, image_arr[:, 0], [1, 1, 1, 1, 1])
    y_fit = f_4(x, popt[0], popt[1], popt[2], popt[3], popt[4])

    for x in range(delta, row, delta):
        gx = abs(y_fit[x - delta] - y_fit[x]) / delta
        gradient[x, 0] = gx  # + gy
        # for y in range(1):
        #     gx = abs(moon_f[x - delta, y] - moon_f[x, y]) / delta
        #     # gy = abs(moon_f[x, y + 1] - moon_f[x, y])
        #     gradient[x, y] = gx # + gy

    plt.subplot('311')
    plt.plot(gradient[:, 0])
    plt.subplot('312')
    plt.plot(y_fit)
    plt.subplot('313')
    plt.imshow(image, cmap='gray')
    plt.show()
    # sharp = moon_f + gradient
    # sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))
    #
    # gradient = gradient.astype("uint8")
    # sharp = sharp.astype("uint8")
    # cv2.imshow("moon", image_arr)
    # cv2.imshow("gradient", gradient)
    # cv2.imshow("sharp", sharp)
    # cv2.waitKey()
    return
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
    travel_dirs = ['cancer', 'normal', 'hsil']
    prefix = ['normal', 'HSIL', 'cancer']
    for idx, td in enumerate(travel_dirs):
        inner_dir = os.path.join(save_dir, travel_dirs[idx])
        tiff_paths = [tp for tp in os.listdir(inner_dir) if tp.endswith('.png')]
        for tp in tiff_paths:
            print(tp)
            save_name = prefix[idx] + "-" + tp[:-5] + '.png'
            feature_ana(os.path.join(inner_dir, tp), os.path.join(save_dir, save_name))
            break
        break

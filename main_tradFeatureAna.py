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
    return A*x*x*x*x + B*x*x*x + C*x*x + D*x + E


def df_4 (x, A, B, C, D, E):
    return A * x * x * x * 4 + B * x * x * 3 + C * x * 2 + D;


def feature_ana(tiff_path, save_path, delta = 1):
    plt.figure(figsize=(5,5))
    # tif = TiffFile(tiff_path)
    image = cv2.imread(tiff_path, 0)[340:, :]
    mean, std = np.mean(image), np.std(image)
    image = cv2.medianBlur(image, 7)
    image_arr = np.mean(image[:, 2210:2220], axis=1)
    print('shape:{}'.format(image_arr.shape))
    row, column = math.ceil(image_arr.shape[0])-delta, 10
    image_arr = image_arr.astype("float")

    gradient = np.zeros((row-1, column))

    x = np.arange(0, image_arr.shape[0], 1)
    popt, pcov = optimize.curve_fit(f_4, x, image_arr, [1, 1, 1, 1, 1])
    y = f_4(x, popt[0], popt[1], popt[2], popt[3], popt[4])
    y_fit = df_4(x, popt[0], popt[1], popt[2], popt[3], popt[4])

    for x in range(delta, row):
        gx = (y[x] - y[x - delta]) / delta
        gradient[x-1, 0] = gx
        # for y in range(1):
        #     gx = abs(moon_f[x - delta, y] - moon_f[x, y]) / delta
        #     # gy = abs(moon_f[x, y + 1] - moon_f[x, y])
        #     gradient[x, y] = gx # + gy

    plt.subplot('311')
    plt.plot(gradient[:, 0])
    plt.plot(y_fit)
    plt.plot(range(0, 600, 20), [0]*30, marker='.', markersize=0.01, linestyle=':')
    plt.title("cancer, avg: {}--std: {}".format(int(mean), round(std, 2)))
    plt.subplot('312')
    plt.plot(y)
    plt.plot(image_arr)
    plt.subplot('313')
    plt.imshow(image, cmap='gray')
    plt.savefig('./results/hsil2-avg-2210-2220.png')
    plt.show()
    print(gradient[:,0])
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
    plt.imshow(image_arr[1].T[:, :], cmap="gray")
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
            # feature_ana(os.path.join(inner_dir, tp), os.path.join(save_dir, save_name))
            feature_ana('./images/hsil/B1903834-11_circle_5.0x5.0_C01_S0009_0.png',
                        os.path.join(save_dir, save_name))
            break
        break

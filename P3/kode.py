from sqlite3 import Row
from cv2 import imshow
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from array import *

# Main Function


def flip_image():
    img = mpimg.imread('img/kuda.jpg')
    horizontal_img = cv2.flip(img, 1)
    vertical_img = cv2.flip(img, 0)
    both_img = cv2.flip(img, -1)

    plt.subplot(2, 2, 1), plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(horizontal_img)
    plt.title('Flip Horizontal'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(vertical_img)
    plt.title('Flip Vertikal'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(both_img)
    plt.title('Flip both'), plt.xticks([]), plt.yticks([])
    plt.show()
    return


def enchancement():
    img = mpimg.imread('img/kuda.jpg')
    row, col, n = img.shape
    img1 = np.zeros((row, col, 3), np.uint8)
    img2 = np.zeros((row, col, 3), np.uint8)
    img3 = np.zeros((row, col, 3), np.uint8)
    img4 = np.zeros((row, col, 3), np.uint8)
    img5 = np.zeros((row, col, 3), np.uint8)

    th = 50
    for y in range(0, col-1):
        for x in range(0, row-1):
            R, G, B = img[x, y]
            if (R+th) > 255:
                R = 255
            else:
                R = R+th
            if (G+th) > 255:
                G = 255
            else:
                G = G+th
            if (R+th) > 255:
                R = 255
            else:
                R = R+th
            img1[x, y] = [R, G, B]

    th = 4
    for y in range(0, col-1):
        for x in range(0, row-1):
            R, G, B = img[x, y]
            if (R*th) > 255:
                R = 255
            else:
                R = R*th
            if (G*th) > 255:
                G = 255
            else:
                G = G*th
            if (R*th) > 255:
                R = 255
            else:
                R = R*th
            img2[x, y] = [R, G, B]

    xmax = 0
    xmin = 300

    for y in range(0, col-1):
        for x in range(0, row-1):
            R, G, B = img[x, y]
            gray = int((R+G+B)/3)
            if(gray > xmax):
                xmax = gray
            if(gray < xmin):
                xmin = gray

    d = xmax-xmin
    for y in range(0, col-1):
        for x in range(0, row-1):
            R, G, B = img[x, y]
            gray = int((R+G+B)/3)
            gray = int((255/d)*gray-xmin)
            img3[x, y] = [gray, gray, gray]

    print("xmax=", xmax)
    print("xmin=", xmin)

    titles = ['Original Image', 'BRIGHTNESS', 'CONTRAST', 'AUTO SCALE']
    images = [img, img1, img2, img3]
    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return


# ======Main Program=======
flip_image()
# enchancement()

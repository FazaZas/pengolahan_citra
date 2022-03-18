import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from array import *


def load_image():
    img = cv2.imread('img/spongebob1.png')
    frame_resize = rescaleFrame(img, scale=5)
    cv2.imshow('arsip', img)
    cv2.imshow('arsip resize', frame_resize)
    cv2.waitKey(0)
    return


load_image()

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from array import *

# Main Function

def load_image():
     img01=cv2.imread('img/A1.jpg')
     img02=cv2.imread('img/A2.jpg')
     img03=cv2.imread('img/A3.jpg')
     img04=cv2.imread('img/A4.jpg')
     
     plt.subplot(2,2,1),plt.imshow(img01)
     plt.title('satu'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,2),plt.imshow(img02)
     plt.title('dua'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,3),plt.imshow(img03)
     plt.title('tiga'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,4),plt.imshow(img04)
     plt.title('empat'), plt.xticks([]), plt.yticks([])
     
     plt.show()
     plt.close()
     return

def load_image1():
     img=cv2.imread('img/arsip.jpeg')
     frame_resize=rescaleFrame(img,scale=5)
     cv2.imshow('arsip',img)
     cv2.imshow('arsip resize',frame_resize)
     cv2.waitKey(0)
     return

def access_image():
     img01 = cv2.imread('img/kuda.jpg')
     row1,col1,n=img01.shape
     print (row1,col1)
     img02 = np.zeros((row1,col1,3), np.uint8)
     img03 = np.zeros((140,200,3), np.uint8)
     img04 = np.zeros((140,200,3), np.uint8)
     
     img02 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
     img03 = img02.copy()
     
     color=(0, 0,255)
     img04 = np.full((140,200,3), color, np.uint8)
     row4,col4,n=img04.shape
     print (row4,col4)
     
     plt.subplot(2,2,1),plt.imshow(img01)
     plt.title('Kuda 01'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,2),plt.imshow(img02)
     plt.title('Kuda 02'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,3),plt.imshow(img03)
     plt.title('Kuda 03'), plt.xticks([]), plt.yticks([])
     plt.subplot(2,2,4),plt.imshow(img04)
     plt.title('Kuda 04'), plt.xticks([]), plt.yticks([])
     plt.show()
     
     return
    


#======Main Program=======
load_image()
# load_image1()
# access_image()
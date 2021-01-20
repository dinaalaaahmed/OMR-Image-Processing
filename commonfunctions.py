import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
from skimage import io
from skimage.morphology import binary_closing, disk
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle
import cv2
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from skimage.color import rgb2gray,rgb2hsv
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.filters import median
from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank, threshold_local
from skimage.filters import try_all_threshold, threshold_triangle, threshold_yen

from skimage.util import img_as_ubyte
import matplotlib
import matplotlib.pyplot as plt
import cv2
from cv2 import threshold 
from collections import Counter 
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening, skeletonize, thin

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

#Model
from sklearn.neural_network import MLPClassifier
import os
import pickle

#Path
import glob 
from pathlib import Path

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

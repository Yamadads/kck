#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage import data, io, filters, morphology, feature, exposure
from matplotlib import pyplot as plt
import numpy as np

files = ['samolot01.jpg','samolot05.jpg','samolot07.jpg',
         'samolot08.jpg','samolot10.jpg','samolot17.jpg']

def getBlackAndWhiteImage(imageArray, riffle):
    height, width = np.shape(imageArray)
    for i in range(height):
        for j in range(width):
            if imageArray[i,j]>riffle:
                imageArray[i,j] = 1
            else:
                imageArray[i,j] = 0
    return imageArray

def getContourImage(imageFile):
    planeImage = data.imread(imageFile,True)
    imageArray = np.asarray(planeImage)
    averageColor = np.mean(imageArray)
    imageArray = getBlackAndWhiteImage(imageArray,averageColor*0.85)
    imageArray = filters.sobel(imageArray)
    imageArray = morphology.dilation(imageArray,morphology.disk(3))
    return imageArray

def paint():
    fig = plt.figure(facecolor='black')
    for i in range(0,len(files)):
        contourImage = getContourImage(files[i])
        ax = fig.add_subplot(2,3,i)
        ax.set_xticks([])
        ax.set_yticks([])
        io.imshow(contourImage)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    paint()
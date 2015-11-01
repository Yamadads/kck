#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage import data, io, filters, morphology, feature, exposure, measure
from matplotlib import pyplot as plt
import numpy as np

files = ['samolot00.jpg','samolot01.jpg','samolot02.jpg','samolot03.jpg',
         'samolot05.jpg','samolot07.jpg',
         'samolot08.jpg','samolot09.jpg','samolot10.jpg','samolot11.jpg',
         'samolot12.jpg','samolot13.jpg','samolot14.jpg','samolot15.jpg',
         'samolot16.jpg','samolot18.jpg','samolot19.jpg','samolot20.jpg',]

def getBlackAndWhiteImage(imageFile):
    image = data.imread(imageFile,True)
    imageArray = np.asarray(image)
    riffle =  (1 - np.mean(imageArray))*0.64
    height, width = np.shape(imageArray)
    for i in range(height):
        for j in range(width):
            imageArray[i,j] = 1-imageArray[i,j]
            imageArray[i,j]=imageArray[i,j]**5
            if imageArray[i,j]> riffle:
                imageArray[i,j] = 1
            else:
                imageArray[i,j] = 0
    imageArray = morphology.closing(imageArray, morphology.square(25))
    imageArray = morphology.dilation(imageArray,morphology.disk(16))
    imageArray = morphology.erosion(imageArray,morphology.disk(10))
    return imageArray

def getCentroid(points):
    x = [p[1] for p in points]
    y = [p[0] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid

def getContours(blackAndWhiteImage,limitContourLength):
    contours = measure.find_contours(blackAndWhiteImage,0.9)
    contours = [contour for contour in contours if len(contour)>limitContourLength]
    return contours

def paint(backgroundColor,centroidColor,contourLineWidth,limitContourLength):
    fig = plt.figure(facecolor=backgroundColor)
    for i in range(0,len(files)):
        originalImage = data.imread(files[i])
        blackAndWhiteImage = getBlackAndWhiteImage(files[i])
        contours = getContours(blackAndWhiteImage,limitContourLength)
        centroids = [getCentroid(contour) for contour in contours]
        ax = fig.add_subplot(4,5,i)
        ax.set_yticks([])
        ax.set_xticks([])
        io.imshow(originalImage)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=contourLineWidth)
        for centroid in centroids:
            ax.add_artist(plt.Circle(centroid,5,color=centroidColor))
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    paint(backgroundColor='black',centroidColor='white',contourLineWidth=2,limitContourLength=375)

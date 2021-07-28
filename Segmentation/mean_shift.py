from sklearn.cluster import MeanShift, estimate_bandwidth
# import sklearn.
import numpy as np
import cv2
import os
from skimage.color import rgb2lab

import matplotlib.pyplot as plt
import argparse

images = []
folder = 'test'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    # im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is not None:
        images.append(img)
i=0
for originImg in images:
    # Shape of original image
    originShape = originImg.shape
    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities
    flatImg=np.reshape(originImg, [-1, 3])
    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    # Performing meanshift on flatImg
    ms.fit(flatImg)
    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels=ms.labels_
    # print("labels:",labels)
    # Remaining colors after meanshift
    cluster_centers = ms.cluster_centers_
    # print("cluster_centers:",cluster_centers)
    # Finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    # Displaying segmented image
    # segmentedImg = np.reshape(labels, originShape[:2])
    # cv2.imshow('Image',segmentedImg)
    # plt.imshow(segmentedImg)
    # plt.show()
    # image = cv2.cvtColor(originImg, cv2.COLOR_RGB2GRAY)
    # segmentedImg = segmentedImg.reshape(image.shape)
    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    # image1 = rgb2lab(segmentedImg)
    plt.imshow(segmentedImg)
    plt.show()
    # cv2.imwrite("Mean-shift_res/{}.jpg".format(i), segmentedImg)
    # i += 1
    # cv2.destroyAllWindows()

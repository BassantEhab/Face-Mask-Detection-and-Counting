import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

images = []
folder = 'test'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    # im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is not None:
        images.append(img)
i=0
for image in images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    print(pixel_values.shape)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image)
    plt.show()
    # cv2.imwrite("K-mean_res/{}.jpg".format(i), segmented_image)
    # i+=1
    # cv2.destroyAllWindows()
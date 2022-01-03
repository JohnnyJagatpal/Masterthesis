"""
    Funktion for watersehd segmentation
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, color,io




def segmentation(img,pixelsize):
    
    # convert image into grayscale if needed and inverse
    if len(np.array(img).shape) == 3:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img1 = img

    kernel = np.ones((3,3), np.uint8)

    # convert into binary with OTSU threshold
    _, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # erode to highlight contours
    erode = cv2.erode(thresh,kernel,iterations=2)

    opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel, iterations=1)

    sure_bg = cv2.dilate(opening,kernel,iterations=2)

    distance_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

    
    _, sure_fg = cv2.threshold(distance_transform,0.025*distance_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unkown = cv2.subtract(sure_bg,sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 10
    
    markers[unkown==225] = 0

    #markers = cv2.watershed(img,markers)

    #img1[markers == -1] = [0,255,255]
    final = color.label2rgb(markers, bg_label=0)


    return final




from matplotlib import pyplot as plt

path = "../../Datasets/PreSampleImages/"
image_name = "000000000603.png"



img = cv2.imread(path + image_name)
ret = segmentation(img, 0.5)
cv2.imwrite('image.png',ret)

plt.imshow(img, cmap='gray')
plt.title('Orginal')
plt.show()
plt.imshow(ret, cmap='gray')
plt.title('New')
plt.show()
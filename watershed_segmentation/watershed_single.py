    """Segmentation of a single image
    """


from matplotlib import pyplot as plt
import cv2
import ultis.watershed as watershed

path = "../Datasets/PreSampleImages/"
image_name = "000000000603.png"



img = cv2.imread(path + image_name)
ret = watershed.segmentation(img, 0.5)


plt.imshow(ret, cmap='gray')
plt.title('Picture')
plt.show()
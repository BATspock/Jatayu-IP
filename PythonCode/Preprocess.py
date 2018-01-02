import cv2
import numpy as np 

class Preprocessing:
    def __init__(self, image):
        self.im = image

    def GaussinaBlur(self):#blur or smoothening helps in denoising and improving edge detection
        return(cv2.GaussianBlur(self.im, (5,5), 0))

    def kmeans(self, K, img): #apply kmeans to find significant colors 5 clusters work good for our targets 
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        return(res.reshape((img.shape)))
    
    def threshold(self, im):#threshold to remove background and and improve contour detection the value of 180 and 255 was found by trial and error work pretty sweet
        ret, thresh = cv2.threshold(im, 180, 255, 0)
        return(ret, thresh)
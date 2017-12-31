#PART 0---------------------------------------------------------------
#import all dependencies and library
import numpy as np
import cv2

#PART 1---------------------------------------------------------------
#resize the image   
image = cv2.imread('im5.jpg')
r= 750.0/image.shape[1]
dim = (750, int(image.shape[0]*r))
img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

#PART 2---------------------------------------------------------------
#apply k means for color based segmentation

img = cv2.GaussianBlur(img, (5,5), 0)
Z = img.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret, label, center = cv2.kmeans(Z, K,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#rebel mode ON

#try thresholding 
#img0 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
#th = cv2.adaptiveThreshold(img0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#cv2.imshow('threshold', th)
#detect contours in result obtianed from k means

#rebel mode OFF

#PART 3
#find relevant contours in the image 

ret, thresh = cv2.threshold(res2, 180, 255, 0)
edged = cv2.Canny(thresh, 30, 200)
#image, contours, hirerachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


#show images from each part-------------------------------------------

#cv2.imshow('original', image)                                 #part 1
cv2.imshow('resized', img)                                     #part 1
                                  
cv2.imshow('res', res2)                                        #part 2

cv2.imshow('thresh', thresh)                                  #part 3
cv2.imshow('edged' ,edged)                                      #part 3
cv2.waitKey(0)
cv2.destroyAllWindows()
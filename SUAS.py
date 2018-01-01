#PART 0---------------------------------------------------------------
#import all dependencies and library
import numpy as np
import cv2

#PART 1---------------------------------------------------------------
#resize the image   
image = cv2.imread('im14.jpg')
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

#PART 3------------------------------------------------------------------------
#find relevant contours in the image 

ret, thresh = cv2.threshold(res2, 180, 255, 0)
img0 = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY) #as findCountours works only on unit8 single channel
_, contours, _ = cv2.findContours(img0.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (255,0,0), 2)

'''
print(len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))

'''
mask = np.zeros(img.shape, dtype = "uint8")

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y+ h), (255, 255, 255), -1)

img1 = cv2.bitwise_and(img, mask)

cv2.imshow("mask", img1)







#show images from each part-------------------------------------------

#cv2.imshow('original', image)                                 #part 1
#cv2.imshow('resized', img)                                     #part 1
                                  
#cv2.imshow('res', res2)                                        #part 2

#cv2.imshow('thresh', thresh)                                  #part 3
#cv2.imshow('gray', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
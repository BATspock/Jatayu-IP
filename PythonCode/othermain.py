#please comment properly for the love of God or girlfriend, whomever you believe watches your every move, if you are a single atheist die, don't touch my code (now no CSE student is ever going to touch my code LOL)
#Yes I wrote the above because I am jobless and have lots of time to waste
#This is so useless. This is like my 100938409218th attemp to write the code to make it work properly
#so here to all the saddness, unhappiness and all things evil in the world......
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from Preprocess import Preprocessing
from ContourOperations import Contours
from Resize import ResizeImage

#import image
im = cv2.imread('target2.jpg')

#rescale image
object_0 = ResizeImage(im)
#image_00= object_0.rescale()

#apply kmeans to the rescaled image

object_1 = Preprocessing(im)
image_10 = object_1.kmeans(4, im)

#threshold the image obtained after kmeans
ret_11, thresh_11 = object_1.threshold(image_10) 

#find contours in the thresholded image
object_2 = Contours()
contours_20 = object_2.FindContours(thresh_11)
object_2.drawContours(-1, contours_20, im)


cv2.imshow('LOL', im)
#plt.imshow(im)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#please comment properly for the love of God or girlfriend, whomever you believe watches your every move, if you are a single atheist die, don't touch my code (now no CSE student is ever going to touch my code LOL)
#Yes I wrote the above because I am jobless and have lots of time to waste
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Resize import ResizeImage
from ContourOperations import Contours
from Preprocess import Preprocessing
from BackSubtraction import FGExtraction
from CropOut import CropOut

#try to keep the image small and precise
im = cv2.imread('im6.jpg')

#Resize image
object_step_0 = ResizeImage(im)
image_step_00 = object_step_0.rescale()

#find foreground from the thresholded image
object_step_10 = FGExtraction()
image_step_100 = object_step_10.ForeGround(image_step_00)


#take result from above step and apply smoothening
object_step_20 = Preprocessing(image_step_100)
image_step_200 = object_step_20.GaussinaBlur()
#on the smoothend image apply kmeans to find significant colors
image_step_210 = object_step_20.kmeans(5,image_step_100)
#on the image obtained after kmeans apply thresholding to remove background
ret_220, thresh_220 = object_step_20.threshold(image_step_210) 


#find contours 
object_step_30 = Contours()
contour_list_300 = object_step_30.FindContours(thresh_220)
#draw contours on image obtained after rescaling
#object_step_30.drawContours(-1, contour_list_300, image_step_00)

#draw bounding rectangle around the detected shape contour
#hopefully it should be the contour with the biggest area in the thresholded image
object_step_40 = CropOut(thresh_220, contour_list_300)
#image to give the bounding rectangle around the target
image_step_400 = object_step_40.BigmakeRect()
#increase the size of obtained image after cropping
object_step_41 = ResizeImage(image_step_400)
image_step_410 = object_step_41.IncreaseSize()

#cv2.imshow('rescaled', image_step_00)
cv2.imshow('thresh',image_step_410)
#cv2.imshow('kmeans', image_step_210)
cv2.waitKey(0)
cv2.destroyAllWindows()
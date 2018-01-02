#please comment properly for the love of God or girlfriend, whomever you believe watches your every move, if you are a single atheist die, don't touch my code (now no CSE student is ever going to touch my code LOL)
#Yes I wrote above because I am jobless and have lots of time to waste
import cv2
import numpy as np 
from Resize import ResizeImage
from ContourOperations import Contours
from Preprocess import Preprocessing
from BackSubtraction import FGExtraction

#try to keep the image small and precise
im = cv2.imread('im12.jpg')

#Resize image
object_step_0 = ResizeImage(im)
image_step_00 = object_step_0.rescale()

#take result from above step and apply smoothening
object_step_10 = Preprocessing(image_step_00)
image_step_100 = object_step_10.GaussinaBlur()
#on the smoothend image apply kmeans to find significant colors
image_step_110 = object_step_10.kmeans(5,image_step_100)
#on the image obtained after kmeans apply thresholding to remove background
ret_120, thresh_120 = object_step_10.threshold(image_step_110) 

#find foreground from the thresholded image
object_step_20 = FGExtraction()
image_step_200 = object_step_20.ForeGround(thresh_120)

#find contours using the foregorund extracted image
object_step_30 = Contours()
contour_list_300 = object_step_30.FindContours(image_step_200)
#draw contours on image obtained after rescaling
object_step_30.drawContours(-1, contour_list_300, image_step_00)

cv2.imshow('rescaled', image_step_00)
#cv2.imshow('blurred', image_step_100)
#cv2.imshow('kmeans', image_step_110)
cv2.waitKey(0)
cv2.destroyAllWindows()
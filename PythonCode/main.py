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
im = cv2.imread('im12.jpg')

#Resize image
object_0 = ResizeImage(im)
image_00 = object_0.rescale()

#find foreground 
object_1 = FGExtraction()
image_step_10 = object_1.ForeGround(image_00)

#find contours 
object_2 = Contours()
contour_20 = object_2.FindContours(image_step_10)
#draw contours on image obtained after rescaling
#object_step_30.drawContours(-1, contour_list_300, image_step_00)

#draw bounding rectangle around the detected shape contour
#hopefully it should be the contour with the biggest area in the thresholded image
object_3 = CropOut(image_00, contour_20)
#image to give the bounding rectangle around the target
image_30 = object_3.BigmakeRect()

#increase the size of obtained image after cropping
object_4 = ResizeImage(image_30)
image_40 = object_4.IncreaseSize()

#blur image 
object_5 = Preprocessing(image_40)
image_50 = object_5.GaussinaBlur()
#apply kmeans
image_51 = object_5.kmeans(5,image_50)
#on the image obtained after kmeans apply thresholding to remove background
ret_52, thresh_52 = object_5.threshold(image_51)
#find contour in the cropped out image
contour_53 = object_2.FindContours(thresh_52)#using the object from part 2 to find and draw contours on the cropped image 
#draw contours in the cropped out image
object_2.drawContours(-1, contour_53, image_40)


cv2.imshow('contour', image_40)
cv2.imshow('thresh', thresh_52)

cv2.waitKey(0)
cv2.destroyAllWindows()
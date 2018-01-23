#please comment properly for the love of God or girlfriend, whomever you believe watches your every move, if you are a single atheist die, don't touch my code (now no CSE student is ever going to touch my code LOL)
#Yes I wrote the above because I am jobless and have lots of time to waste
#This is so useless. This is like my 1009384092180980th attemp to write the code to make it work properly
#so here to all the saddness, unhappiness and all things evil in the world......
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from Preprocess import Preprocessing
import ContourOperations as co
from Resize import ResizeImage
from CropOut import CropOut

#import image
im = cv2.imread('PICT_20180118_175217.JPG')
#resize image to 1/2th to reduce the number of pixel
resize = ResizeImage(im)
target = resize.rescale()#for test images sent by the previous batch
#target = resize.IncreaseSize()#for small size images

#blur image to enhance the target
preprocessing = Preprocessing(target)
im0 = preprocessing.GaussinaBlur()#blur images to identify edges easily and remove noise in backgournd ......kernel size is 9X9 
im1 = preprocessing.kmeans(8, im0)#apply kmeans to help remove background 

#use canny to find edges
im2 = cv2.Canny(im1, 80, 255)

#find coutours for other relevent operations 
l = co.FindContours(im2)
#make the rectanle around the biggest contour
rect = CropOut(target, l)
im3 = rect.BigmakeRect()
#draw contours........this is optional
#contours.drawContours(-1, l, target)

#increase size of cropped image
resize_later = ResizeImage(im3)
final = resize_later.IncreaseSize()

#apply kmeans to reduce the number of colors in final image
im4 = preprocessing.kmeans(4, final)

#find edges in target image
im5 = cv2.Canny(im4, 150, 255)

#find contours in the target image after canny
l_target = co.FindContours(im5)
co.drawContours(-1, l_target, final)

#cv2.imshow('kmeans', im1)
#cv2.imshow('target', target)
#cv2.imshow('thresh', im3)
#cv2.imshow('identify', im2)
cv2.imshow('final', im5)
cv2.imshow('r', im4)
cv2.imshow('new', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
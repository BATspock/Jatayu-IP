import cv2
import numpy as np 

def findContours(im):
        #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #im is thresholded image obtained from Preprocess after applying threshold function
        _,contours, _ = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return (contours)
        
def drawContours(contNo, contour, img):#draw contours on original image, contour obtained from FindContours, contNo = the contour to be displayed
        cv2.drawContours(img, contour, contNo, (255,0,0), 2) #draws blue lines around contours detected
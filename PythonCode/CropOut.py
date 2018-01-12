import cv2
import numpy as np 

class CropOut:#contour is the contour list obtained after 
    def __init__(self, image, contour):
        self.im = image
        self.con = contour
        self.c = -1
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        #find the contour with the biggest area that should be hopefully the required contour 
        if len(contour)!=0:
            self.c = max(contour, key = cv2.contourArea)#store the biggest contour in c
            
    def BigIndex(self):
        return(self.con.index(self.c))#return index of the biggest contour

    def BigmakeRect(self):#return image with bounding rectangle along the biggest contour
        self.x, self.y, self.w, self.h = cv2.boundingRect(self.c)
        #return(cv2.rectangle(self.im,(x,y),(x+w, y+h),(0,255,0),2))
        return(self.im[self.y:self.y+self.h, self.x:self.x+self.w])
    
     
import cv2
#create class to resize an image
class ResizeImage:
    def __init__(self, image):
        self.img = image

    #method to rescale the image to 1/2 th without loss of color    
    def rescale(self):
        newx, newy = int(self.img.shape[1]/2),int(self.img.shape[0]/2)
        return(cv2.resize(self.img, (newx, newy)))
        

    def IncreaseSize(self):#incrase size of image by 5 times
        newx, newy = int(5*self.img.shape[1]),int(5*self.img.shape[0])
        return(cv2.resize(self.img, (newx, newy)))
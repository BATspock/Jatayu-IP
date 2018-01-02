import cv2
#create class to resize an image
class ResizeImage:
    def __init__(self, image):
        self.img = image

    #method to rescale the image to 1/4 th without loss of color    
    def rescale(self):
        newx, newy = int(self.img.shape[1]/4),int(self.img.shape[0]/4)
        newimg = cv2.resize(self.img, (newx, newy))
        return newimg


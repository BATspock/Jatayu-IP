import cv2
import numpy as np 
import matplotlib.pyplot as plt
from Resize import ResizeImage

class FGExtraction:#used foreground extraction to extract the target from backgrounds
    def ForeGround(self, image):
        mask = np.zeros(image.shape[:2], np.uint8)

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        rect = (75,75, int(2*image.shape[1]/3), int(2*image.shape[0]/3))

        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
        image = image*mask2[:,:, np.newaxis]
        return image


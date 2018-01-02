import cv2
import numpy as np 
import matplotlib.pyplot as plt
from Resize import ResizeImage

class FGExtraction:
    def ForeGround(self, image):
        mask = np.zeros(image.shape[:2], np.uint8)

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        rect = (150, 150, 350, 250 )

        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
        image = image*mask2[:,:, np.newaxis]
        return image


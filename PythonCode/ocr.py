from PIL import Image
import cv2
import pytesseract as tsrct
from tesserocr import PyTessBaseAPI, RIL, PyPageIterator, PyLTRResultIterator, iterate_level
from Preprocess import Preprocessing
import ContourOperations as contrOps
from Resize import ResizeImage
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


############################################################
# NOT USED 

def rotate_box(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb
#################################################################################

def rotate_and_detect(targetImg):

    predList = []
    scoreList = []
    with PyTessBaseAPI() as api:
        for theta in range(0, 0, 10):
            rotatedImg = rotate_bound(targetImg, theta)
            cv2.imwrite(str(theta)+".png", rotatedImg)
            img = str(theta)+".png"
            
            api.SetImageFile(img)
            api.SetPageSegMode(10)
            api.Recognize()
            
            ri = api.GetIterator()
            level = RIL.SYMBOL
            for r in iterate_level(ri, level):
                symbol = r.GetUTF8Text(level) 
                conf = r.Confidence(level)
                if symbol:
                    print("symbol: {}, conf: {}".format(symbol, conf))
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()


















def ocr(targetImg):
    # convert to gray and threshold
    greyImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
    threshImg = cv2.threshold(greyImg, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    
    rows, cols, width = targetImg.shape
    
    contrList = contrOps.findContours(threshImg)
    contrList.sort(key = lambda s: -len(s))
    top3 = contrList[0:2]
    predList = []
    i=0
    
    for contr in top3:
        # orientation
        center, axis,angle = cv2.fitEllipse(contr)
        print(angle)
        x_shift = (cols/2) - center[0]
        y_shift = (rows/2) - center[1]
        x, y, w, h = cv2.boundingRect(contr)

        M = np.float32([[1,0,x_shift],[0,1,y_shift]])
        dst1 = cv2.warpAffine(threshImg,M,(cols,rows))
        cv2.imwrite("lol" +str(i) + ".png", dst1)
        rows1, cols1 = dst1.shape
        x1 = int(x + x_shift)
        y1 = int(y + y_shift)
        w = int(w)
        h = int(h)
        bb = [[x1,y1], [x1+w,y1], [x1+w, y1+h], [x1, y1+h]]
        image2 = np.zeros((rows1, cols1), np.int8)
        mask = np.array([bb], dtype=np.int32)
        cv2.fillPoly(image2, [mask],255)
        maskimage2 = cv2.inRange(image2, 1, 255)
        output = cv2.bitwise_and(dst1, dst1, mask=maskimage2)
        cv2.imwrite("lololol" +str(i) + ".png", output)
        dst2 = rotate_bound(output, 180 - angle)
        cv2.imwrite("lel" +str(i) + ".png", dst2)
        rows2, cols2 = dst2.shape
        print((cols2, rows2))
        predChar = tsrct.image_to_string(Image.open("lel" +str(i) + ".png"), config='-psm 10')
        print(predChar)
        if predChar.isalnum():
            predList.append(predChar)
        i = i+1
    
    return angle

print(ocr(cv2.imread("im15.jpg")))
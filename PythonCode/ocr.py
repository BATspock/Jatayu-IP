from PIL import Image
import cv2
import pytesseract as tsrct
from Preprocess import Preprocessing
import ContourOperations as contrOps


def ocr(targetImg):
    # convert to gray and threshold
    greyImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
    threshImg = cv2.threshold(greyImg, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    
    contrList = contrOps.findContours(threshImg)
    contrList.sort(key = lambda s: -len(s))
    top3 = contrList[0:2]
    predList = []
    for contr in top3:
        x, y, w, h = cv2.boundingRect(contr)
        targetCrop = threshImg[y:y+h, x:x+h]
        ip = Preprocessing(targetCrop)
        blurImg = ip.GaussinaBlur()
        cv2.imwrite("blurTester.png", blurImg)
        predChar = tsrct.image_to_string(Image.open("blurTester.png"), config='-psm 10000')
        if predChar.isalnum():
            predList.append(predChar)

    return predList

print(ocr(cv2.imread("crop1.jpeg")))




















'''
# add file path
filePath = "/Users/Starck/Jatayu-IP/PythonCode/crop1.jpeg"
image = cv2.imread(filePath)

ip = Preprocessing(image)
kMeansImg = ip.kmeans(2, image)
cv2.imshow("kMeansImg", kMeansImg)

# convert to gray and threshold
greyImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshImg = cv2.threshold(greyImg, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

#display processed images
cv2.imshow("ThresholdedImage", threshImg)

#save to disk
threshFile = "tester.png"
cv2.imwrite(threshFile, threshImg)

ip = Preprocessing(cv2.imread(threshFile))
blurImg = ip.GaussinaBlur()
cv2.imshow("BlurredImage", blurImg)
cv2.imwrite("blurTester.png", blurImg)
#run through ocr engine
text = tsrct.image_to_string(Image.open("blurTester.png"), config='-psm 10000')
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
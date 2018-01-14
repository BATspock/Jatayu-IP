from PIL import Image
import cv2
import pytesseract as tsrct
# add file path
filePath = "crop1.jpeg"
image = cv2.imread(filepath)

# convert to gray and threshold
greyImg = cv2.cvtColor(image, cv2.COLORBGR2GRAY)
threshImg = cv2.threshold(greyImg, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

#display processed images
cv2.imshow("Thresholded Image", threshImg)

#save to disk
threshFile = "tester.png"
cv2.imwrite(filename, threshImg)

#run through ocr engine
text = tsrct.image_to_string(Image.open(threshFile))
print(text)




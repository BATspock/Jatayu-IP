from PIL import Image
import cv2
import pytesseract as tsrct
# add file path
filePath = "/Users/Starck/Jatayu-IP/PythonCode/crop3.jpeg"
image = cv2.imread(filePath)

# convert to gray and threshold
greyImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshImg = cv2.threshold(greyImg, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

#display processed images
cv2.imshow("ThresholdedImage", threshImg)

#save to disk
threshFile = "tester.png"
cv2.imwrite(threshFile, threshImg)

#run through ocr engine
text = tsrct.image_to_string(Image.open(threshFile))
print(text)




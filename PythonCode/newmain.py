#please comment properly for the love of God or girlfriend, whomever you believe watches your every move, if you are a single atheist die, don't touch my code (now no CSE student is ever going to touch my code LOL)
#Yes I wrote the above because I am jobless and have lots of time to waste
#This is so useless. This is like my 1009384092180980th attemp to write the code to make it work properly
#so here to all the saddness, unhappiness and all things evil in the world......
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from Preprocess import Preprocessing
import ContourOperations as co
from Resize import ResizeImage
from CropOut import CropOut
from PIL import Image
import pytesseract as tsrct
from BackSubtraction import FGExtraction


#import image
im = cv2.imread('/home/aditya/suas/PICT_20180212_173618.JPG')
#resize image to 1/2 to reduce the number of pixel
resize = ResizeImage(im)
target = resize.rescale()#for test images sent by the previous batch do 1/4th and new camera
#target = resize.IncreaseSize()#for small size images
#target = im
#blur image to enhance the target
preprocessing = Preprocessing(target)
im0 = preprocessing.GaussinaBlur()#blur images to identify edges easily and remove noise in backgournd ......kernel size is 9X9 
im1 = preprocessing.kmeans(8, im0)#apply kmeans to help remove background 

#use canny to find edges
im2 = cv2.Canny(im1, 80, 255)

#find coutours for other relevent operations 
l = co.FindContours(im2)

#make the rectanle around the biggest contour//////........main logic of the code
rect = CropOut(target, l)
im3 = rect.BigmakeRect()
#draw contours........this is optional
#contours.drawContours(-1, l, target)

#increase size of cropped image
resize_later = ResizeImage(im3)
final = resize_later.IncreaseSize()

#apply kmeans to reduce the number of colors in final image
im4 = preprocessing.kmeans(3, final)

#im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
#creare another image to find external contour
imc = preprocessing.kmeans(2, final)
#find edges in target image
im5 = cv2.Canny(im4, 150, 255)
#find edges in the external contour
imc1 = cv2.Canny(imc, 150, 255)
#find contours in the target image after canny
l_target = co.FindContours(im5)

conts_1 = sorted(l_target, key = cv2.contourArea)

conts = conts_1[::-1][:4]

'''
-circle
semicircle
quarter_circle
-triangle
square
rectangle
trapezoid
-pentagon
-hexagon
-heptagon
-octagon
-star
-cross
'''

#detect shapes using number contour  
for c in conts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015*peri, True)
    if (len(approx)==6):
        screenCnt = approx
        print('hexagon')
        break
    if (len(approx) == 4):
        screenCnt = approx
        print('square')
        break
    if (len(approx)== 3):
        screenCnt = approx
        print('triangle')
        break
    if (len(approx)== 5):
        print('pentagon')
        screenCnt = approx
        break
    if (len(approx)== 7):
        print('heptagon')
        screenCnt = approx
        break
    if (len(approx) == 8):
        print('octagon')
        screenCnt = approx
        break
    if (len(approx)== 10):
        print('star')
        screenCnt = approx
        break
    if (len(approx)== 12):
        print('cross')
        screenCnt = approx

        break
    if (len(approx)>=60):
        print('circle')
        screenCnt = approx
        break

#print(screenCnt)
cv2.drawContours(final, [screenCnt], -1, (0, 255, 0), 2)

req_conts=[]
for c in conts_1:
    if cv2.contourArea(c)>=100:
        req_conts.append(c)


contsAreanew = list(cv2.contourArea(c) for c in req_conts)
from CropOut import innerRect
imsomething = innerRect(im4, req_conts[0])
cv2.imshow('letter', imsomething)


gray = cv2.cvtColor(imsomething, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
colors = np.where(hist>100)
img_number = 0
for color in colors[0]:
    print(color)
    split_image = imsomething.copy()
    split_image[np.where(gray != color)] = 0
    cv2.imwrite(str(img_number)+".jpg",split_image)
    img_number+=1

'''
plt.hist(gray.ravel(),256,[0,256])
plt.savefig('plt')
plt.show()



# cv2.imshow('mask', np.hstack([im4, output]))
#cv2.imwrite('mask.jpg', output)
im6 = cv2.blur(output,(5,5))
im6_1 = cv2.blur(im6, (5,5))
cv2.imshow('kmeans', im6_1)
cv2.imwrite('finalkmeans.jpg', im6_1)
'''
#cv2.imwrite('kmeansimg.jpg', im4)
#cv2.imshow('target', target)
#cv2.imshow('thresh', im3)
cv2.imshow('identify', imc)
cv2.imshow('final',final)
#cv2.imshow('r', im3)
cv2.imshow('new', im4)
cv2.waitKey(0)
cv2.destroyAllWindows()

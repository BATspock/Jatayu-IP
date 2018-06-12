def ipcode(path):   
    #please comment properly for the love of God or girlfriend, whomever you believe watches your every move, if you are a single atheist die, don't touch my code (now no CSE student is ever going to touch my code LOL)
    #Yes I wrote the above because I am jobless and have lots of time to waste
    #This is so useless. This is like my 1009384092180980th attemp to write the code to make it work properly
    #so here to all the saddness, unhappiness and all things evil in the world......
    import cv2
    import numpy as np 
    import matplotlib.pyplot as plt 
    from Preprocess import GaussianBlur, kmeans, threshold
    from ContourOperations import FindContours, drawContours
    from Resize import rescale, IncreaseSize
    from CropOut import CropOut
    from PIL import Image
    import pytesseract as tsrct
    #resize image to 1/2 to reduce the numberr of pixel
    im = cv2.imread(path)

    target = rescale(im)#for test images sent by the previous batch do 1/4th and new camera
    #target = resize.IncreaseSize()#for small size images
    #target = im
    #blur image to enhance the target
    im0 = GaussianBlur(target)#blur images to identify edges easily and remove noise in backgournd ......kernel size is 9X9 
    im1, _= kmeans(8, im0)#apply kmeans to help remove background 

    #use canny to find edges
    im2 = cv2.Canny(im1, 80, 255)

    #find coutours for other relevent operations 
    l = FindContours(im2)

    #make the rectangle around the biggest contour//////........main logic of the code
    rect = CropOut(target, l)
    im3,x, y = rect.BigmakeRect()
    #draw contours........this is optional
    #contours.drawContours(-1, l, target)
    
    #increase size of cropped image
    final = IncreaseSize(im3)

    #apply kmeans to reduce the number of colors in final image
    im4,centroids = kmeans(3, final)
    cv2.imshow('check1', im4)
    print(centroids[1], centroids[2])

    #creare another image to find external contour
    imc,_ = kmeans(2, im4)
    cv2.imshow('check', imc)
    #find edges in target image
    im5 = cv2.Canny(im4, 150, 255)
    im5 = cv2.blur(im5, (3,3))
    #find edges in the external contour
    imc1 = cv2.Canny(imc, 150, 255)
    imc1 = cv2.blur(imc1, (3,3))
    cv2.imshow('check1', imc1)


    #find contours in the target image after canny
    l_target = FindContours(im5)
    #print(hir)

    conts_1 = sorted(l_target, key = cv2.contourArea)
    #cv2.drawContours(im4, l_target, -1, (0,0,255), 3)

    conts = conts_1[::-1][:3]

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
    i = -1 
    shape = ""
    for c in conts:
        i += 1
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05*peri, True)
        if (len(approx)==6):
            screenCnt = approx
            shape = "hexagon"
            break
        if (len(approx) == 4):
            screenCnt = approx
            shape = "square"
            break
        if (len(approx)== 3):
            screenCnt = approx
            shape = "triangle"
            break
        if (len(approx)== 5):
            screenCnt = approx
            shape = "pentagon"
            break
        if (len(approx)== 7):            
            screenCnt = approx
            shape = "heptagon"
            break
        if (len(approx) == 8):            
            screenCnt = approx
            shape = "octagon"
            break
        if (len(approx)== 10):
            screenCnt = approx
            shape = "star"
            break
        if (len(approx)== 12):
            screenCnt = approx
            shape = "cross"
            break
        if (len(approx)>=15):
            screenCnt = approx
            shape = "circle"
            break


    #cv2.drawContours(final, [screenCnt], -1, (0, 255, 0), 2)
    M = cv2.moments(conts[i])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    #geolocation maths
    final_x = 2*(cx + (x/5.0))
    final_y = 2*(cy + (y/5.0))

    #mask image to get only inside of the contour
    cv2.imshow('yolo',im4)
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(im4)
    cv2.drawContours(mask, conts, i, 255, -1)
    out = np.zeros_like(im4)
    out[mask == 255] = im4[mask == 255]

    newout = cv2.Canny(out, 50, 200)
    newout = cv2.blur(newout, (3,3))
    cv2.imshow('newout', newout)

    contsL = FindContours(newout)
    contsLnew = sorted(contsL, key= cv2.contourArea)
    l= list (cv2.contourArea(c) for c in contsLnew)

    contsnewreq = []
    for c in contsLnew:
        if cv2.contourArea(c)>=0.15*(max(l)):
            contsnewreq.append(c)
    #print(len(contsnewreq))

    mask1 = np.zeros_like(im4)
    cv2.drawContours(mask1, contsnewreq, 0, 255, -1)
    out1 = np.zeros_like(im4)
    out1[mask1 == 255] = im4[mask1 == 255]
    cv2.imshow('newoutletter', out1)

    #cv2.imshow('latest', im0)
    #cv2.imshow('identify', im1)
    #cv2.imshow('final',im2)
    #cv2.imshow('r', im3)
    #cv2.imshow('new', im4)
    #cv2.imshow('im5', im5)
    #cv2.imshow('imc1', imc1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (centroids[1], centroids[2], out1, shape, final_x, final_y)

val0, val1, ocr, shape, geox, geoy = ipcode('/home/aditya/suas/PICT_20180212_173318.JPG') 
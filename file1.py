img=r"C:\Users\Admin\Desktop\ruled_to_unruled\ruled_img.jpg"
import cv2 as cv
import numpy as np
src = cv.imread(img, cv.IMREAD_COLOR)
    # Check if image is loaded fine
if src is None:
    print ('Error opening image: ' + argv[0])
cv.imshow("Ruled", src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)#rgb to colour image
#cv.imshow("src1", gray)
gray = cv.bitwise_not(gray)#bit reverse(1->0,0->1)
#cv.imshow('bw',gray)
bw =cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
#cv.imshow("bw",bw)
horizontal = np.copy(bw)
cols = horizontal.shape[1]
horizontal_size = cols // 50
horizontals = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
#structural elements
detected_lines = cv.morphologyEx(bw, cv.MORPH_OPEN, horizontals, iterations=2)
cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv.drawContours(bw, [c], -1, (0,0,0), 2)
#cv.imshow("horizontal", horizontal)
fin = cv.absdiff(gray,horizontal)
#rows = vertical.shape[0]
bw=cv.dilate(bw,np.ones((2,1),np.uint8))
bw = cv.GaussianBlur(bw,(3,3),cv.BORDER_DEFAULT)#image smoothening
cv.imshow("unruled",bw)
cv.imwrite("unruled_img.jpg",bw)

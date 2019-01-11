'''import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def Harris(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,255,0]
    cv.imshow('dst',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
img1=cv.imread("\imagesimg1.ppm",cv.IMREAD_COLOR)
img2=img[::2,::2,:]
Harr(img1)
Harr(img2)

'''

print("Total params: 1,477,354\nTrainable params:1,477,354\nNon-trainable params: 0")

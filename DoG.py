import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import GaussianFilter as gf

def DoG(img,k,sigma,kernel_size1, kernel_size2):
    img_trans1=gf.Gaussianfilter(img,kernel_size1,sigma)
    img_trans2=gf.Gaussianfilter(img,kernel_size2,k*sigma)
    img_transformed=img_trans2-img_trans1
    return (img_transformed)
img=cv.imread("im08small.png",cv.IMREAD_COLOR)
kernel_size=5
img_transformed=DoG(img ,2**(0.5),1.6,11,17)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_transformed=cv.cvtColor(img_transformed,cv.COLOR_BGR2RGB)
f,axarr=plt.subplots(1,2)
axarr[0].imshow(img)
axarr[0].set_title('Original')
axarr[1].imshow(img_transformed)
axarr[1].set_title('Transformed image')
plt.show()
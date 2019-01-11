import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def gaussianFilter(img ,kernel_size,sigma):
    p=kernel_size//2
    img_transformed = np.zeros(img.shape)
    kernel=np.zeros((kernel_size,kernel_size))
    kernel1=np.zeros((1,kernel_size))
    kernel2=np.zeros((kernel_size,1))
    for t in range(kernel_size):
        x=t-p
        kernel1[0][t]=(np.exp(-((x**2)/(2*(sigma**2))))/np.sqrt(2*np.pi*(sigma**2)))
        kernel2[t] = (np.exp(-((x**2)/(2*(sigma**2))))/np.sqrt(2*np.pi*(sigma**2)))
    kernel=np.matmul(kernel2,kernel1)
    kernel_sum=np.sum(kernel)
    kernel=kernel/kernel_sum
    a, b = img.shape[0], img.shape[1]
    for i in range(3):
        for x in range(p, a - p):
            for y in range(p, b - p):
                region = img[x - p:x + p + 1, y - p:y + p + 1, i]
                region_sum=np.sum(np.multiply(region,kernel))
                img_transformed[x, y, i] = region_sum
    img_transformed = img_transformed.astype('uint8')
    return img_transformed
img=cv.imread("images\im07small.png",cv.IMREAD_COLOR)
kernel_size=5
img_transformed=gaussianFilter(img ,5,1.8)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_transformed=cv.cvtColor(img_transformed,cv.COLOR_BGR2RGB)
f,axarr=plt.subplots(1,2)
axarr[0].imshow(img)
axarr[0].set_title('Original image')
axarr[1].imshow(img_transformed)
axarr[1].set_title('Sigma of transformed image : '+str(1.8)+"\nKernel size: "+str(5))
plt.show()

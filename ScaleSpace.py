import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def gaussianFilter(img ,kernel_size,sigma):
    w=kernel_size//2
    img_transformed = np.zeros(img.shape)
    kernel=np.zeros((kernel_size,kernel_size))
    kernel1=np.zeros((1,kernel_size))
    kernel2=np.zeros((kernel_size,1))
    for t in range(kernel_size):
        x=t-w
        kernel1[0][t]=(np.exp(-((x**2)/(2*(sigma**2))))/np.sqrt(2*np.pi*(sigma**2)))
        kernel2[t] = (np.exp(-((x**2)/(2*(sigma**2))))/np.sqrt(2*np.pi*(sigma**2)))
    kernel=np.matmul(kernel2,kernel1)
    kernel_sum=np.sum(kernel)
    kernel=kernel/kernel_sum
    a, b = img.shape[0], img.shape[1]
    for x in range(w, a - w):
            for y in range(w, b - w):
                region = img[x - w:x + w + 1, y - w:y + w + 1]
                region_sum=np.sum(np.multiply(region,kernel))
                img_transformed[x, y] = region_sum
    #img_transformed=img_transformed[w:a-w,w:b-w]
    # cv.imshow("Gaussian filered",img_transformed)
    # cv.waitKey(0)
    return img_transformed
import math
def DoG(img,k,sigma):
    if(math.ceil(6*sigma)%2)==1:
       k1=math.ceil(6*sigma)
    else:
        k1 = (math.ceil(6*sigma))+1
    if (math.ceil(6*k*sigma) % 2) == 1:
        k2 = math.ceil(6*k*sigma)
    else:
        k2 = (math.ceil(6*k*sigma)) + 1
    img_trans1=gaussianFilter(img,k1,sigma)
    img_trans2=gaussianFilter(img,k2,k*sigma)
    img_transformed=img_trans2-img_trans1
   # cv.imshow("DoG", img_transformed)
    cv.waitKey(0)
    return (img_transformed)
def ScaleSpace(img,k,sigma0,nims):
    keypoints=[]
    img_mid = np.zeros(img.shape)
    downs=1
    s1,s2=img.shape
    while(max(s1,s2)>=(2**0.5)*sigma0*(k**4)):
        sigma = sigma0
        layers=[]
        for var3 in range(nims):
           layers.append(DoG(img,k,sigma))
           sigma=k*sigma
        for var1 in range(layers[0].shape[0]):
                for var2 in range(layers[0].shape[1]):
                    for var3 in range(1,nims-1):
                            points = []
                            for varx in range(max(var1 - 1, 0), min(var1 + 2, layers[0].shape[0])):
                                for vary in range(max(var2 - 1, 0), min(var2 + 2, layers[0].shape[1])):
                                    if(varx,vary)!=(var1,var2):
                                        points.append(layers[var3-1][varx, vary])
                                        points.append(layers[var3][varx, vary])
                                        points.append(layers[var3+1][varx, vary])
                            if (layers[var3][var1, var2] >max(points) or layers[var3][var1, var2] < min(points)):
                                keypoints.append((( downs*var1, downs*var2), downs*( sigma0*(k**var3))))
        sigma0=sigma0*(k**2)
        img=img[::2,::2]
        s1=s1//2
        s2=s2//2
        downs=downs*2


    img1=cv.imread("img1.ppm",cv.IMREAD_COLOR)
    print(len(keypoints))
    print(img1.shape[0]*img1.shape[1])
    for keyp in keypoints:
        cv.circle(img1, (keyp[0][1],keyp[0][0]), int(keyp[1]*(2**0.5)),(0,255,0))
    cv.imshow("Image",img1)
    cv.waitKey(0)
img=cv.imread("img1.ppm",cv.IMREAD_GRAYSCALE)
kernel_size=5
ScaleSpace(img,1.5,1.8,4)


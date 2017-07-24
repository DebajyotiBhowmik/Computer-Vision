#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:10:20 2017

@author: debajyoti
"""

import numpy as np

import numpy as np
import cv2
from scipy.stats import multivariate_normal
import scipy.misc

def transform_image(img,prob,name):
    new_img=np.zeros(img.shape)
    total_no_colm=img.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j]=prob[total_no_colm*(i)+j]*(img[i][j][0])
            
    scipy.misc.imsave(name+'.jpg', new_img)    

def find_gaussian(list_of_skin_image):
   
    list_of_train_vector=[]
    for img_name in list_of_skin_image:
        img=cv2.imread(img_name)
        converted=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
        for i in range(converted.shape[0]):
            for j in range(converted.shape[1]):
                list_of_train_vector.append([converted[i][j][1]/50,converted[i][j][2]/50]) 
                     
    mean=np.mean(list_of_train_vector,axis=0)
    print('mean=')
    print(mean)
    matrix=np.asarray(list_of_train_vector).T
    covariance=np.cov(matrix)
    print('cov')
    print(covariance)
    dist=multivariate_normal(mean=mean,cov=covariance)
    return dist
def image_show(img):
    #img = cv2.imread('input.png', 0)
    kernel = np.ones((5,5), np.uint8)

    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    
    cv2.imshow('Input', img)
    cv2.imshow('Erosion', img_erosion)
    cv2.imshow('Dilation', img_dilation)
    cv2.waitKey(0)
    
"""img1=cv2.imread('skin1.jpg',cv2.COLOR_RGB2YCrCb)

converted=cv2.cvtColor(img1,cv2.COLOR_RGB2YCrCb)
cv2.imshow('d',converted[:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows()"""

list_of_skin_image=['skin1.jpg','skin2.jpg']#,'skin3.jpg','skin4.jpg','skin5.jpg','skin6.jpg','skin7.jpg','skin8.jpg','skin9.jpg','skin10.jpg','skin11.jpg','skin12.jpg','skin13.jpg','skin14.jpg','skin15.jpg','skin16.jpg','skin17.jpg','skin18.jpg']
dist=find_gaussian(list_of_skin_image)
print(dist)

img=cv2.imread('newface.jpg')


img=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
list_of_test_vector=[]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        list_of_test_vector.append([img[i][j][1]/50,img[i][j][2]/50])
        
prob=dist.pdf(list_of_test_vector)
transform_image(img,prob,'changed-newface')
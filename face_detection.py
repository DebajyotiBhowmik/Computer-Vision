#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:03:20 2017

@author: debajyoti
"""

import numpy as np
import cv2
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

def image_show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img= cv2.imread('changed-face6.jpg')
image_show(img[:,:,0])

file_name=['changed-face1.jpg','changed-face2.jpg','changed-face3.jpg','changed-face4.jpg','changed-face5.jpg','changed-face6.jpg','changed-face9.jpg','changed-face10.jpg','changed-face12.jpg','changed-face13.jpg','changed-face14.jpg','changed-face15.jpg','changed-face16.jpg']

list_of_train_img=[]

for name in file_name:
    img= cv2.imread(name)
    list_of_train_img.append(np.ndarray.flatten(img[:,:,0]))
  
pca = PCA(n_components=10)   
 
pca.fit(list_of_train_img)  

list_of_components=pca.components_
###########window size 49 pixel

new_img= cv2.imread('changed-face8.jpg')
new_img=np.ndarray.flatten(new_img[:,:,0])

Y=pca.transform(new_img)
new_img_approx=pca.inverse_transform(Y)


diff=np.subtract(new_img,new_img_approx)  

for i in range(diff.shape[1]):
    if diff[0][i]<0:
        diff[0][i]=0
        
conv_img=np.reshape(diff,(720,1280))  
      
image_show(conv_img)        
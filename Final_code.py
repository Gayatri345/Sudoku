#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from matplotlib import pyplot
import os


# In[17]:


img = mpimg.imread('pic13.png')
height=img.shape[0]
width=img.shape[1]
channels=img.shape[2]
cell_size=round(height/9)
def cropImage(image):
    crop_list=[]
    i=0
    x= cell_size
    
    while True:
        j=0
        y=cell_size
        while True:
            cropped_img2 = img[i*cell_size+10:x-10,j*cell_size+10:y-10]
            pyplot.imsave('cell'+str(i)+str(j)+'.png',cropped_img2)
            j=j+1
            y=y+cell_size
            if(j>=9):
                break 
        i=i+1
        x=x+cell_size
        if(i>=9):
            break
        
cropImage(img)

def BlankToZero():
    cell_list=[]
    for i in range(0,9):
        for j in range(0,9):
            image='cell'+str(i)+str(j)+'.png'
            cell_list.append(image)
    for cell in cell_list:
        file_size = os.stat(cell)
    
    empty_images = []
    for image in cell_list:
        file_size = os.stat(image)
        if file_size.st_size == 372:
            empty_images.append(image)
    for cell in empty_images:
        img1=mpimg.imread('image0.png')
        img = mpimg.imread(cell)
        pyplot.imsave(cell,img1)
        img = mpimg.imread('cell01.png')
BlankToZero()


# In[ ]:





# In[ ]:





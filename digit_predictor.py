#!/usr/bin/env python
# coding: utf-8

# In[114]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import cv2
import matplotlib.pyplot as plt


# #Training Model with MNIST data set

# In[167]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
X_train_flatten = X_train.reshape(len(X_train), 28*28)
X_test_flatten = X_test.reshape(len(X_test), 28*28)
model = keras.Sequential([keras.layers.Dense(100, input_shape =(784,),activation = 'relu'),
                         keras.layers.Dense(10, activation = 'sigmoid')])
#opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train_flatten, y_train, epochs = 10,batch_size=32)
model.evaluate(X_test_flatten, y_test)
y_predicted = model.predict(X_test_flatten)
y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[168]:


model.evaluate(X_test_flatten, y_test)


# #Getting Images of 0 to 1

# In[169]:


cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
cm


# In[170]:

'''
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('predicted')
plt.ylabel('truth')
'''

# In[171]:


image_list=[]
for i in range(0,10):
    image='image'+str(i)+'.png'
    image_list.append(image)
#print('image_list: ',image_list)


# #Predicting images from 0 to 9

# In[172]:


#images 0 to 9 prediction
import time
predict_list=[]
for image in image_list:
    #print(image)
    file= image
    #print(file)
    test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)
    img2_resized= img_resized/255
    #print(img_resized)
    img_resized_flatten = img2_resized.reshape(1, 28*28)
    y_predicted = model.predict(img_resized_flatten)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    #print(y_predicted_labels)
    predict_list.append(y_predicted_labels)
    #time.sleep(3)
print("\n \n List of predicted numbers from 0 to 1")
print(predict_list)
    


# #Test prediction for any single image

# In[173]:


file= 'image3.png' #test image
#print(file)
test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)
img2_resized= img_resized/255
#print(img_resized)
img_resized_flatten = img2_resized.reshape(1, 28*28)
y_predicted = model.predict(img_resized_flatten)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

#print(y_predicted_labels)


# #Reading the images of Sudoku grid(81 cell images) in to list

# #Reading images of SUDOKU GRID

# In[203]:

image_list=[]
for i in range(0,9):
    for j in range(0,9):
        image='cell'+str(i)+str(j)+'.png'
        image_list.append(image)
#print('image_list: ',image_list)


# #predicting given set of images and appending it to a list using our trained model

# In[204]:


import time
predict_list=[]
for image in image_list:
    #print(image)
    file= image
    #print(file)
    test_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(test_image, (28,28), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)
    img2_resized= img_resized/255
    #print(img_resized)
    img_resized_flatten = img2_resized.reshape(1, 28*28)
    y_predicted = model.predict(img_resized_flatten)
    y_predicted_labels = [np.argmax(i) for i in y_predicted]
    #print(y_predicted_labels)
    predict_list.append(y_predicted_labels)
    #time.sleep(3)
    
#print(predict_list)


# In[205]:


my_array = np.array(predict_list)
puzzle = np.reshape(my_array, (9,9))
print('\n Predicted sudoku input matrix\n \n puzzle=\n \n',puzzle)

# Original input image matrix
#  [[5 0 3 0 0 0 0 0 0]
#  [2 0 0 3 0 0 0 0 0]
#  [0 4 0 7 1 0 2 0 3]
#  [0 0 5 4 0 0 0 7 1]
#  [0 0 4 2 0 1 8 0 0]
#  [6 8 0 0 0 7 5 0 0]
#  [1 0 7 0 6 9 0 3 0]
#  [0 0 0 0 0 4 0 0 6]
#  [0 0 0 0 0 0 9 0 5]]

# sudoku input matrix
#  [[5 0 8 0 0 0 0 0 0]
#  [2 0 0 3 0 0 0 0 0]
#  [0 4 0 7 1 0 2 0 3]
#  [0 0 5 4 0 0 0 7 1]
#  [0 0 9 2 0 1 8 0 0]
#  [6 8 0 0 0 7 5 0 0]
#  [5 0 7 0 6 9 0 9 0]
#  [0 0 0 0 0 4 0 0 5]
#  [0 0 0 0 0 0 3 0 5]]
# #with learning_rate

# sudoku input matrix
#  [[5 0 3 0 0 0 0 0 0]
#  [2 0 0 3 0 0 0 0 0]
#  [0 4 0 7 1 0 2 0 3]
#  [0 0 5 4 0 0 0 7 1]
#  [0 0 4 2 0 1 8 0 0]
#  [6 8 0 0 0 7 5 0 0]
#  [1 0 7 0 6 9 0 3 0]
#  [0 0 0 0 0 4 0 0 6]
#  [0 0 0 0 0 0 9 0 5]]
# for picture 50

# sudoku input matrix
#  [[5 0 3 0 0 0 0 0 0]
#  [2 0 0 3 0 0 0 0 0]
#  [0 4 0 7 1 0 2 0 3]
#  [0 0 5 4 0 0 0 7 1]
#  [0 0 4 2 0 1 8 0 0]
#  [6 8 0 0 0 7 5 0 0]
#  [1 0 7 0 6 9 0 3 0]
#  [0 0 0 0 0 4 0 0 6]
#  [0 0 0 0 0 0 3 0 5]
#  
# for picture22

# #Plotting original image

# In[248]:



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from matplotlib import pyplot
img = mpimg.imread('pic13.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





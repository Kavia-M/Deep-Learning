#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install tensorflow


# # IMPORT NECESSARY PACKAGES 

# In[7]:


import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


# # LOAD MNIST DATASET INTO KERAS

# In[8]:


objects =  tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = objects.load_data()


# ### ANALYSING THE DATASET

# In[9]:


for i in range(9):
   # define subplot
   plt.subplot(330 + 1 + i)
   # plot raw pixel data
   plt.imshow(training_images[i])


# In[10]:


print(training_images.shape)
print(training_images[100])


# Above we can see that pixel ranges from 0 to 255

# # BUILD THE NETWORK

# The network bolow is built with 1 input layer, 1 hidden layer with 128 neurons and 1 output layer with 10 neurons

# In[11]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), 
                                    tf.keras.layers.Dense(128, activation='relu'), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# # RESHAPE IMAGE ARRAY INTO [0,1] INTERVAL (normalization)

# In[12]:


training_images  = training_images / 255.0
test_images = test_images / 255.0


# In[14]:


print(training_images[100])


# # DEFINE COMPILER PARAMETERS LIKE OPTIMIZER AND METRICS

# In[15]:


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# # FEED THE DATA TO OUR NETWORK AND TRAIN

# In[16]:


model.fit(training_images, training_labels, epochs=7)


# # TEST THE PERFORMANCE WITH TEST IMAGE

# In[17]:


print(model.evaluate(test_images,test_labels))


# Accuracy 97.85%

# ### Visualizing our prediction using some sample test images

# In[32]:


from matplotlib.pyplot import imshow, show  


# In[36]:


prediction=model.predict(test_images)
for i in range(100,110):
    print(np.argmax(prediction[i]))
    plt.figure(figsize = (10,1))
    plt.imshow(test_images[i])
    show()


# In[ ]:





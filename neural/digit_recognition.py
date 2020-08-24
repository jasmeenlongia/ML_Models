import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd
data=pd.read_csv("train.csv")

x=data.iloc[:,1:]
y=data.iloc[:,0]

from sklearn.model_selection import train_test_split
train_images,test_images,train_labels,test_labels= train_test_split(x,y,test_size=0.3, random_state=100)

train_images.shape #29400 examples with 784 pixels

train_images=train_images.to_numpy()
train_images=train_images.reshape(29400,28,28)
test_images=test_images.to_numpy()
test_images=test_images.reshape(12600,28,28)

train_labels=train_labels.to_numpy()
test_labels=test_labels.to_numpy()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#scaling the data to be btw 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(512, activation='sigmoid'),
    keras.layers.Dense(256, activation='sigmoid'),
    keras.layers.Dense(128, activation='sigmoid'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)

#making predictions
predictions = model.predict(test_images)
#This method returns to us an array of predictions for each image we passed it

np.argmax(predictions[0]) # to get the value with the highest score we can use a useful function from numpy called argmax().
#This simply returns the index of the maximium value from a numpy array.
              
          

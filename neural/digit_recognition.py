import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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
              
          

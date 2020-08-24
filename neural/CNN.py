import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

mnist = datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from tensorflow import keras
#building the Convolutional Base
model = keras.Sequential([keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                         keras.layers.MaxPooling2D((2, 2)),
                         keras.layers.Conv2D(64, (3, 3), activation='relu'),
                         keras.layers.MaxPooling2D((2, 2)),
                         keras.layers.Conv2D(64, (3, 3), activation='relu')])
                         
#Now take these extracted features and classify them
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)


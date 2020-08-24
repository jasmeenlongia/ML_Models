# loading the IMDB movie review dataset from keras. 
# This dataset contains 25,000 reviews from IMDB where each one is already preprocessed and has a label 
# as either positive or negative. Each review is encoded by integers that represents how common a word is in the entire dataset. 
# For example, a word encoded by the integer 3 means that it is the 3rd most common word in the dataset.
import tensorflow as tf

from tensorflow.keras import datasets
import tensorflow.keras.preprocessing.sequence as s
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584 #number of words
MAXLEN = 250 #length of the review we want to have
BATCH_SIZE = 64

mnist=datasets.imdb
(train_data, train_labels), (test_data, test_labels) = mnist.load_data(num_words = VOCAB_SIZE)

#every train data i.e. review has different length
# We cannot pass different length data into our neural network.
#Therefore, we must make each review the same length
train_data = s.pad_sequences(train_data, MAXLEN)
test_data = s.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), #embedding 88584 words into vectors of 32 dimensions
    tf.keras.layers.LSTM(32), #accepts vectors of 32 dimensions
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

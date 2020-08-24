import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()

categories = data.target_names
train = fetch_20newsgroups(subset='train',categories=categories)
test = fetch_20newsgroups(subset='test',categories=categories)
print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline 

model = make_pipeline(TfidfVectorizer(),MultinomialNB()) #pushes data from tfidfVectorizer into the naive bayes
model.fit(train.data, train.target)  #target are the categories; train.data is mapped to TfidfVectorizer() and then goes to nb whick decides the category
predictions = model.predict(test.data)

from sklearn.metrics import confusion_matrix
confusion_matrix(test.target,predictions)

from sklearn.metrics import accuracy_score
accuracy_score(test.target,predictions)

def predict_category(s,train=train, model=model):
    pred=model.predict([s]) #pushes string s into model pipeline
    return categories[pred[0]]
    
predict_category('cost discount price buy sell')

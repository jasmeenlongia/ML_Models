import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data=pd.read_csv("play_or_no.txt")
x=data.drop(['play'],axis=1)
y=data.iloc[:,3]

outlook_type=pd.get_dummies(data['outlook'])
x['humidity_high']=x.humidity.map({'high':1,'normal':0})
x['wind_speed']=x.wind.map({'strong':1,'weak':0})
y=pd.get_dummies(data['play'],drop_first=True)

x=pd.concat([x,outlook_type],axis=1)
x=x.drop(['outlook','humidity','wind'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3, random_state=100)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

predictions=logmodel.predict(x_test)
accuracy_score(y_test,predictions)

from sklearn import tree
clf=tree.DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(x_train,y_train)

predictions2=clf.predict(x_test)
accuracy_score(y_test,predictions2)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=0)
clf.fit(x_train,y_train)

predictions3=clf.predict(x_test)
accuracy_score(y_test,predictions3)

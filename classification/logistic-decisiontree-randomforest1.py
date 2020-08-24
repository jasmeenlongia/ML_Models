import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data=pd.read_csv("titanic.txt")
x=data.drop(['survival'],axis=1)
y=data.iloc[:,3]
x.head(5)

sex=pd.get_dummies(data['gender'],drop_first=True) //we change the gender from male/female to a column that gives 0 for female and 1 for male
x=pd.concat([x,sex],axis=1)
x=x.drop(['gender'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=1) //diving data into test and train datasets

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

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

df=pd.read_csv("fuelconsumption")
df.head()

#CHECK RELATION OF EVERY ATTRIBUTE, MUST BE LINEAR
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS,  color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] // selecting only the features that we require

x=cdf.drop(['CO2EMISSIONS'],axis=1) //choosing independent variables
y=cdf.iloc[:,3] //selecting dependant variable

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3, random_state=1) //dividing the data into training and testing sets

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit (x_train, y_train) // training the model

predictions=regr.predict(x_test)//making predictions for test data set

//calculating the accuracy provided by the model
from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((predictions - y_test) ** 2))
print("R2-score: %.2f" % r2_score(predictions , y_test) )

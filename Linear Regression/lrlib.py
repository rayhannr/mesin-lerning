import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #regeressor = machine
regressor.fit(X, y) #fit = learning the training set how to predict the salary based on exp
#fit itu nyari b0 dan b1

# Predicting the Test set results
y_pred = regressor.predict(X) #apply hasil fit xtrain shg salary = b0 + b1 x exp pada xtest

# Visualising the Training set results
plt.scatter(X, y, color = 'red') #param : x axis, y axis, warna observation value
plt.plot(X, regressor.predict(X), color = 'blue') #hasil regresi
plt.show()
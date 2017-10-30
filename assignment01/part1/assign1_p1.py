#Assignment 1 - Question 1      Akshit Arora
#still working on the plotting the hyperplane on 3D matplotlib

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

#reading data
print "Reading Data"
dataframe = pd.read_csv('assign1_data.txt', delim_whitespace=True)
x_values = dataframe[['x1', 'x2']]
y_values = dataframe[['y']]

#train model on data
print "Training linear model (ordinary least squares) on data."
reg = linear_model.LinearRegression()
reg.fit(x_values, y_values)

#report values
print "w1 = " + str(reg.coef_[0][0])
print "w2 = " + str(reg.coef_[0][1])
print "b = " + str(reg.intercept_[0])

#calculating error
Z = reg.predict(x_values)
print "Mean squared error: %.2f" % mean_squared_error(y_values,Z)

''' #working on it!
#getting results
X1 = np.arange(-3.0,3.0)
X2 = np.arange(-3.0,3.0)
#Z = reg.predict(X)
Z = X1 * reg.coef_[0][0]+ X2 * reg.coef_[0][1]+ reg.intercept_[0]

#visualizing results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X1,X2,Z,color='blue')
ax.scatter(dataframe[['x1']],dataframe[['x2']],y_values, c='red', marker = 'o')
ax.set_xlabel('x axis')
ax.set_xlabel('y axis')
ax.set_xlabel('z axis')
#plt.show()
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

#constants used throughout the program. Feel free to experiment with these
num_of_samples = 100 #number of samples
bound = 4 #upper bound of random interval is bound * pi
deg = 5 #degree of polynomial to regress with

#get a random sample of floats, stored in a numpy array
X = np.random.uniform(low=0, high=bound * np.pi, size=(num_of_samples,))
X = np.sort(X)
print("The first few entries of X:")
print(X[:10])

#feed random sample to the Sin function to create the dataset
y = np.sin(X)
print("The first few entries of y:")
print(y[:10])

poly = PolynomialFeatures(degree=deg, include_bias=False)
poly_X = poly.fit_transform(X.reshape(-1,1))
print("The first few entries of poly_X:")
print(poly_X[:10])

#create training and test data
X_train, X_test, y_train, y_test = train_test_split(poly_X, y, test_size=.25, random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))


#polynomial regression
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)

#test on test data
y_pred = poly_reg_model.predict(X_test)
print("Test set predictions:\n{}".format(y_pred[:10]))
print("MSE on test set: {:.2f}".format(MSE(y_test,y_pred)))

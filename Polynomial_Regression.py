#importing the libraries                          
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values
Y=dataset.iloc[:,-1].values
#training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression 
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
#training the polynoimial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression() 
lin_reg_2.fit(X_poly,Y)
#visualizing linear regression results (for comparison)
'''                   
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or Bluff ( LinearRegression )")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
'''
#visualizing polynomial regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

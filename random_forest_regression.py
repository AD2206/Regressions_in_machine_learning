import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')



X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values 

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=10)
regressor.fit(X, y)

print(regressor.predict([[6.5]]))

X_grid = np.arange(X.min(), X.max(), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1)) 
r2 = regressor.score(X, y)
print(r2)


plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()

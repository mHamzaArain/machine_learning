# Data Preprocessing Template

# Step 1: Importing the libraries
import numpy as np               # to inculude mathematics
import matplotlib.pyplot as plt  # Ploting graph
import pandas as pd              # import & manage datasets 

# Step 2: Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset.info()
dataset.describe()

X = dataset.iloc[:, 1].values.reshape((-1, 1)) # row=all, col=exculude last column
y = dataset.iloc[:, 2].values.reshape((-1, 1)) # row=all, col=3


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(a))

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)  # min X to max x incremented by 0.1
X_grid = X_grid.reshape((len(X_grid), 1)) # row=X_grid.size, col=1
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


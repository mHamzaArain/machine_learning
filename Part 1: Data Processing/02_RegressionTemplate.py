# Step 1: Importing the essential libraries
import numpy as np               # to inculude mathematics
import matplotlib.pyplot as plt  # Ploting graph
import pandas as pd              # import & manage datasets 

# Step 2: Importing & fullfilling prerequisites the dataset 
dataset = pd.read_csv('missing_data.csv')

## Step 2(a): Clearing Data
dataset.describe() # count, mean, std, min, max, 25%, 50%, 75%
dataset.info()	 # names > cols, data type, memory usage
print(dataset.isnull().sum())  # Report missing values

"""
## Step 2(b)(Optional): Fixing missing values & droping col
mean = dataset['CEC'].mean()
dataset['CEC'].fillna(mean, inplace = True)
mean = dataset['phosphorus'].mean()
dataset['phosphorus'].fillna(mean, inplace = True)
mean = dataset['potassium'].mean()
dataset = dataset.drop(['potassium'], axis =1)

print(dataset.isnull().sum())
"""

X = dataset.iloc[:, :-1].values # row=all, col=exculude last column
y = dataset.iloc[:, 3].values # row=all, col=3

"""
##Step 2(b)(Optional): Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',  # missing values which are not available
                  strategy='mean',       # its by default
                  axis=0)                # axis=0 - mean of cols
                                                # axis=1 - mean of rows
imputer = imputer.fit(X[:, 1:3])         # Where to apply, row=all, cols= 2nd(ind-1) & 3rd(ind-2+1)
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Insert average of all rows in NaN area
"""

"""
## Step 2(3): Catagorical data
### Dummy Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # Sampling data in cluster, dummy encoding
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])   # array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0], dtype=object)
### To catagorize col 0 -> country > dummy encoding(binary)(alphabatical order i.e; France, Germany, Spain)
onehotencoder = OneHotEncoder(categorical_features = [0])  # holds the categories expected in the ith column > 0-off, 1-on 
X = onehotencoder.fit_transform(X).toarray()

### Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)   # array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
"""

# Step 3: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,   # dependent, independent variables
                                                    test_size = 0.2, # 20% test size
                                                    random_state = 0) # due to this our trained value is fixed else it may very

"""
# Step 3(a): Features either standardiation or normalization
## Distance b/w age^2 & salary^2 is dominant is salary as it's greater
## Standization = X-mean(X)/S.D(X) 
## Normalization = X-min(x)/(max(x)-min(x))
from sklearn.preprocessing import StandardScaler                # Standrization of dependent variables
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)  
X_test = sc_X.transform(X_test) 
"""

# Step 4: Applying technique & fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Step 5: Predict the result
# Predicting the Test set results
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test)).reshape((-1, 1))
print('\n'.join(''.join(str(cell) for cell in row) for row in y_pred))     # Print predicted values

#################OPTIONAL
# Step 6: Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

############Building Optional model
# Step 7: Final report
from statsmodels import api as sm

X = np.append(np.ones((50, 1)).astype(int), # ones apppending in bigining
              values = X,      # after one's append Values of X
              axis=1)          # apeends in Col
X_opt = X[:, [0,1, 2, 3, 4, 5]] # 1's, Florida, New York, R&D, Administration, Marketing
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()



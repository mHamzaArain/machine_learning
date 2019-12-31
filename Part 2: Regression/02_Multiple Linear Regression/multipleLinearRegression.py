# Step 1: Importing the libraries
import numpy as np               # to inculude mathematics
import matplotlib.pyplot as plt  # Ploting graph
import pandas as pd              # import & manage datasets 

# Step 2: Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # R&D Spend,Administration Spend, Marketing Spend, State
y = dataset.iloc[:, 4].values # Profit


# Step 2(b): Catagorize data
## Dummy Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   # Sampling data in cluster, dummy encoding
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])   # array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0], dtype=object)
# To catagorize col 0 -> country > dummy encoding(binary)(alphabatical order i.e; California, Florida  , Newyork)
onehotencoder = OneHotEncoder(categorical_features = [3])  # holds the categories expected in the ith column > 0-off, 1-on 
X = onehotencoder.fit_transform(X).toarray()

# avoidng dummy variable to resist from dummy variable variable trap
X = X[:, 1:]  # eleminating 1st Col

# Step 3: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,   # dependent, independent variables
                                                    test_size = .20, # 20% test size
                                                    random_state = 0) # due to this our trained value is fixed else it may very

# Step 4: Applying technique
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 5: Predict the result
# Predicting the Test set results
y_pred = regressor.predict(X_test)

############Building Optional model for backward elimination
# y = b0 * (X0) + b1X1 + --- + bnXn
# since, stats model doesnt contain b0 as cont. i.e; we inculde x0 in it above
from statsmodels import api as sm

X = np.append(np.ones((50, 1)).astype(int), # ones apppending in bigining
              values = X,      # after one's append Values of X
              axis=1)          # apeends in Col
X_opt = X[:, [0,1, 2, 3, 4, 5]] # 1's, Florida, New York, R&D, Administration, Marketing
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()  # ols = ordinary Least square
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5]] # Eliminate Florida=1
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]] # Eliminate New York=2
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]] # Eliminate New York=2
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()



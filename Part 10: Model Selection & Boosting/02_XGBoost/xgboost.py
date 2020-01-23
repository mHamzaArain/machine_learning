# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

# Part 1 - Data Preprocessing

# Substep 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Score, Geography, Gender, Tenure, Balance, Products, Card, Active, Salary 
y = dataset.iloc[:, 13].values   # Exited Bank

## Substep 2(a): Catagorical data
### Dummy Encoding the Independent Variable
# Encoding categorical data to one hot encoded form
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Region: Farance=0, Germany=1 & spain=2
labelencoder_region = LabelEncoder()
X[:, 1] = labelencoder_region.fit_transform(X[:, 1]) 

# Gender: Female=0 & Male=1
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2]) 
# holds the categories expected in the ith column > 0-off, 1-on 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).todense() # Encoding 'Geography' to one hot
X = X[:, 1:] # get rid of dummy variable trap

# Substep 3: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,   # dependent, independent variables
                                                    test_size = .2, # 20% test size
                                                    random_state = 0) # due to this our trained value is fixed else it may very

# Part 2: Building odel
# Substep 1: Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

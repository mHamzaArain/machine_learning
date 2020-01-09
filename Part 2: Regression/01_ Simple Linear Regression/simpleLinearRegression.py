## Data Processing

# Step 1: Importing the libraries
import numpy as np               # to inculude mathematics
import matplotlib.pyplot as plt  # Ploting graph
import pandas as pd              # import & manage datasets 

# Step 2: Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # yearsExperience
y = dataset.iloc[:, 1].values # salary


# Step 3: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,   # dependent, independent variables
                                                    test_size = 1/3, # 20% test size
                                                    random_state = 0) # due to this our trained value is fixed else it may very

# # Step 3(a): Features either standardiation or normalization
# ## Distance b/w age^2 & salary^2 is dominant is salary as it's greater
# ## Standization = X-mean(X)/S.D(X) 
# from sklearn.preprocessing import StandardScaler                # Standrization of dependent variables
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X)  
# X_test = sc_X.transform(X_test) 

# Step 4: Applying technique
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 5: Predict the result
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


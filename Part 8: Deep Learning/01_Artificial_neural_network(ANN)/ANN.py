# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

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

# # Substep 3(a): Features either standardiation or normalization
# ## Distance b/w age^2 & salary^2 is dominant is salary as it's greater
# ## Standization = X-mean(X)/S.D(X) 
from sklearn.preprocessing import StandardScaler                # Standrization of dependent variables
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  
X_test = sc_X.transform(X_test) 

# Part 2 - Artificial Neural Network(ANN)

# Substep 1: Importing the Keras libraries and packages
import keras
from keras.models import Sequential    # To init. neural network
from keras.layers import Dense         # To build layers

# Substep 2: Initialising the ANN
classifier = Sequential()       # Defining seq.of layers 

# Substep 3: Building input, hidden & output layers
# Adding the input layer and the first hidden layer
classifier.add(
        Dense(
                output_dim = 6,      # adding no. of nodes > (features(X)=11 + 1)/2 = 6 
                init = 'uniform',    # uniform function to init. wieghts
                activation = 'relu', # relu > rectifier function
                input_dim = 11       # features(X)=11
                ))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform',
                     activation = 'sigmoid'))   # sigmoid function for 1 DV 
                                                # Softmax function: Actually sigmoid function but used for more than 1 DV.
                                                
# Substep 4: Compiling the ANN
classifier.compile(optimizer = 'adam',            # Type of Stochastic gradient algo to organizing of back propagation in most efficient way
                   metrics = ['accuracy'],        # 
                   loss = 'binary_crossentropy')  # O/P is 1 variable (binary)->  binary_crossentropy
                                                    # more than 2 O/P variables called -> catagorical_crossentropy 
                   
                    
# Substep 5: Fitting the ANN to the Training set
classifier.fit(X_train, y_train,    # dependant & independant variables
               batch_size = 25,     # spliting data into 10 
               nb_epoch = 200)      # Process of propagation to loss function is called epoch


# Part 3 - Making the predictions and evaluating the model

# Substep 1: Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)              # 50% threshould > 
                                        # Greater than 50%; True for exit 
                                        # less than 50%; False for to remain in bank

# Substep 2: Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
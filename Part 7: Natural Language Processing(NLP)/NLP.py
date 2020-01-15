#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:29:01 2020

@author: hamza
"""

# Natural Language Processing

# Step 1: Importing the libraries
from ClassificationModels import test
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',
                      delimiter = '\t',         # Due to tsv file thus separated by tab
                      quoting = 3)              # No quoting allowed

# Step 3: Cleaning the texts
def cleaning_text(dataSet):
    import re
    import nltk
    nltk.download('stopwords')        
    from nltk.corpus import stopwords 
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer() 

    corpus = list()    
    for comment in range(dataset['Review'].size):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][comment])   
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# Step 4: Tokenization
from sklearn.feature_extraction.text import CountVectorizer # class for tokenization
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(cleaning_text(dataset)).toarray() 
y = dataset.iloc[:, 1].values 

# Step 5: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,   # dependent, independent variables
                                                    test_size = 0.25, # 25% test size
                                                    random_state = 0) # due to this our trained

# Step 6: Applying Models
test(X_train, X_test, y_train, y_test,
                          naive_bayes=True,
                          decision_tree=True,
                          random_forest=True,
                          SVM=True,
                          KNN=True,
                          logestic_regression=True)











        
        
        
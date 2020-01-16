# Natural Language Processing

# Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',
                      delimiter = '\t',         # Due to tsv file thus separated by tab
                      quoting = 3)              # No quoting allowed

# Step 3: Cleaning the texts
import re
import nltk
nltk.download('stopwords')        # For dowloading words matching 
from nltk.corpus import stopwords    # Importing stopwords for matching in text 
from nltk.stem.porter import PorterStemmer  # For Stemping process
ps = PorterStemmer()                        # Stemp object

corpus = []
comments = dataset['Review'].size # 1000

for comment in range(comments):
    # Substep 1: Eliminating unnecessary characters(i.e; numbers, comma, dot, etc)
    ##  'Wow... Loved this place.' ->  'Wow    Loved this place '
    review = re.sub('[^a-zA-Z]',            # # Not a-z, A-z contained 
                    ' ',                    # # Reaplace anomaly by space
                    dataset['Review'][comment])   # eachi comment from dataset
    
    # Substep 2: Captal to small letters to form uniformity 
    ## 'Wow    Loved this place ' -> 'wow    loved this place ' -> ['wow', 'loved', 'this', 'place']
    review = review.lower()
    review = review.split()
    
    # Substep 3:  
    ## ['wow', 'loved', 'this', 'place'] ->  ['wow', 'loved', 'place'] (word 'this' has been eliminated  
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    # Substep 4:  Stempinng Process (i.e; Converting froms of word into simple) -> Loved/Loving = Love
    ## ['wow', 'loved', 'place']   -> ['wow', 'love', 'place'] -> wow love place
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    
    corpus.append(review)

# Step 4: Creating the Bag of Words model -> Classificatino(zero=negative, 1=positive comment)
# Substep 1: Tokenization(i.e; make column for each word)
from sklearn.feature_extraction.text import CountVectorizer # class for tokenization
cv = CountVectorizer(
        # stop_words='english',  # Remove stop words()i.e; has no sense e.g; from, up, etc)
        # lowercase=True,        # Convert all characters to lowercase before tokenizing.
        # token_pattern='word',  # Regular expression takes only alphabetes
        max_features = 1500    #  build a vocabulary that ordered by term frequency across the corpus.
        )
X = cv.fit_transform(corpus).toarray()  # Sparse matrix(i.e; containing zero high sparsity)
                                        # row=comments,
                                        # col=tkoenizaed for each word(i.e; 0 = Not appear word in that comment
                                                                          # 1 = Appeared word in that comment)
y = dataset.iloc[:, 1].values   # 0=Negative comment, 1=positive comment 

# Step 3: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,   # dependent, independent variables
                                                    test_size = 0.25, # 25% test size
                                                    random_state = 0) # due to this our trained

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

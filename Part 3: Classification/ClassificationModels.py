#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:56:01 2020

@author: hamza
"""
# For Confusion Matrix
from sklearn.metrics import confusion_matrix

class Models:
    """
    By testing the dataset to identify suitable model.
    Total models:
        1. Naive Bayes.
        2. Decision Tree.
        3. Random Forest.
        4. Support Vector Machine(SVM).
        5. K-Nearest Neighbour(KNN).
        6. Logestic Regression.
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Parameters
        ----------
        X_train: np.array
            Independant variable to be trained.
            
        X_test: np.array 
            Independant variable to be testet.
            
        y_train: np.array
            Dependant variable to be trained.
            
        y_test: np.array
            Dependant variable to be tested.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    
    def naive_bayes(self):
        """Apply Naive bayes classification model.
                
        Parameter
        ---------
        None
        
        Return
        ------
        cm[0][0]: float
            True negative
            
        cm[0][1]: float
            False Positive
            
        cm[1][0]: float
            False Negative
            
        cm[1][1]:
            True Positive
            
        cm: np.array contain int
        """
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)
        cm = confusion_matrix(self.y_test, classifier.predict(self.X_test))
        
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1], cm
    
    def decision_tree(self):
        """Apply Decision tree classification model.
        
        Parameter
        ---------
        None
        
        Return
        ------
        cm[0][0]: float
            True negative
            
        cm[0][1]: float
            False Positive
            
        cm[1][0]: float
            False Negative
            
        cm[1][1]:
            True Positive
            
        cm: np.array contain int
        """
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        cm = confusion_matrix(self.y_test, classifier.predict(self.X_test))
        
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1], cm
    
    def random_forest(self):
        """Apply Random forest classification model.
                
        Parameter
        ---------
        None
        
        Return
        ------
        cm[0][0]: float
            True negative
            
        cm[0][1]: float
            False Positive
            
        cm[1][0]: float
            False Negative
            
        cm[1][1]:
            True Positive
            
        cm: np.array contain int
        """
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=1, criterion='entropy', random_state=0)      
        classifier.fit(self.X_train, self.y_train)
        cm = confusion_matrix(self.y_test, classifier.predict(self.X_test))
        
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1], cm
    
    def SVM(self, kernel='rbf'):
        """Apply Support vector machine classification model
        
        Parameter
        ---------
        kernel: (Bydefault; 'rbf').
            Other kernels: 'linear', 'poly' & 'sigmoid'
                
        Parameter
        ---------
        None
        
        Return
        ------
        cm[0][0]: float
            True negative
            
        cm[0][1]: float
            False Positive
            
        cm[1][0]: float
            False Negative
            
        cm[1][1]:
            True Positive
            
        cm: np.array contain int
        """
        from sklearn.svm import SVC
        classifier = SVC(kernel=kernel, random_state=0)
        classifier.fit(self.X_train, self.y_train)
        cm = confusion_matrix(self.y_test, classifier.predict(self.X_test))
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1], cm
    
    def KNN(self):
        """Apply K-nearest Neighbour classification model.
                
        Parameter
        ---------
        None
        
        Return
        ------
        cm[0][0]: float
            True negative
            
        cm[0][1]: float
            False Positive
            
        cm[1][0]: float
            False Negative
            
        cm[1][1]:
            True Positive
            
        cm: np.array contain int
        """
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(self.X_train, self.y_train)
        cm = confusion_matrix(self.y_test, classifier.predict(self.X_test))
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1], cm
    
    def logestic_regression(self):
        """Apply Logestic regression classification model.
                
        Parameter
        ---------
        None
        
        Return
        ------
        cm[0][0]: float
            True negative
            
        cm[0][1]: float
            False Positive
            
        cm[1][0]: float
            False Negative
            
        cm[1][1]:
            True Positive
            
        cm: np.array contain int
        """
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(self.X_train, self.y_train)
        cm = confusion_matrix(self.y_test, classifier.predict(self.X_test))
        return cm[0][0], cm[0][1], cm[1][0], cm[1][1], cm


# ###########################################################################################
    
class MeasurementTools:
    """
    It give multiple tool to mearsure dataset.
        1. Accuracy
        2. Precision
        3. Recall
        4. F1 Score
    """
    def __init__(self, true_negative, false_positive, false_negative, true_positive):
        """ 
        Parameters
        ----------
        true_negative, false_positive, false_negative, true_positive: int 
        """
        self.TN = true_negative
        self.FP = false_positive
        self.FN = false_negative
        self.TP = true_positive
        
    def accuracy(self):
        """Difference of acutal & predicted values.
        
        Parameters
        ----------
        None
        
        Return
        ------
        float
        
        """
        return (self.TN + self.TN) / (self.TP + self.TN + self.FP + self.FN) *100
    
    def precision(self):
        """Measuring exactness.
                
        Parameters
        ----------
        None
        
        Return
        ------
        float
        
        """
        return self.TP / (self.TP + self.FP)
    
    def recall(self):
        """Measuring completeness.
                
        Parameters
        ----------
        None
        
        Return
        ------
        float
        
        """
        return self.TP / (self.TP + self.FN)
    
    def f1_score(self):
        """compromise between Precision and Recall
                
        Parameters
        ----------
        None
        
        Return
        ------
        float
        """
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
        
    
# ##############################################################################################
        
def test(X_train, X_test, y_train, y_test,
                          naive_bayes=False,
                          decision_tree=False,
                          random_forest=False,
                          SVM=False,
                          KNN=False,
                          logestic_regression=False):
    
    """Display results by applying models.
    Total to apply:
        1. Naive Bayes.
        2. Decision Tree.
        3. Random Forest.
        4. Support Vector Machine(SVM) with 4 kernels composed of 
           linear, Gaussian, polynomial, Sigmoid Function.
        5. K-Nearest Neighbour(KNN).
        6. Logestic Regression.
        
    Display result:
        1. True Negative
        2. True Positive
        3. False Negative
        4. Accuracy
        5. Precision
        6. Recall
        7. F1 Score
        
        
    Parameters
    ----------
    X_train, X_test, y_train, y_test: np.array
    decision_tree, random_forest, SVM, KNN, logestic_regression: bool
        (Bydefault: False) to apply modle switch it to true. 
    
    Return
    ------
    None
        
        
    """
    
    classification_model = Models(X_train, X_test, y_train, y_test)
    models = {
          classification_model.naive_bayes: [naive_bayes, "Naive Bayes"],
          classification_model.decision_tree: [decision_tree, "Decision Tree"],
          classification_model.random_forest: [random_forest, "Random Forest"],
          classification_model.SVM: [SVM, "Support Vector Machine"],
          classification_model.KNN: [KNN, "K-Nearest Neighboir"],
          classification_model.logestic_regression: [logestic_regression, "Logestic Regression"]
          }
    
    for model, flag_and_name in models.items():
        if (flag_and_name[0] == True) and (flag_and_name[1] != "Support Vector Machine"):
            TN, FP, FN, TP, cm = model()
            report = MeasurementTools(TN, FP, FN, TP)
            print("\n===================================================================================")
            print(f"Model: {flag_and_name[1]} Result")
            print(f'------------------------------------\n')
            print(f"True Negative: {TN}")
            print(f"True Positive: {TP}")
            print(f"False Negative: {FN}")
            print(f"False Positive: {FP}\n")
            print(f"Accuracy (Difference of acutal & predicted values): {report.accuracy()}")
            print(f"Precision (measuring exactness): {report.precision()}")
            print(f"Recall (measuring completeness): {report.recall()}")
            print(f"F1 Score(compromise between Precision and Recall): {report.f1_score()}")
            print("=====================================================================================\n")
        
        if (flag_and_name[0] == True) and (flag_and_name[1] == "Support Vector Machine"):   
            print("Model: Support Vector Machine Classification Result")
            kernels = ['linear', 'rbf', 'poly', 'sigmoid']

            for kernel in kernels:
                TN, FP, FN, TP, cm = classification_model.SVM(kernel=kernel)
                report = MeasurementTools(TN, FP, FN, TP)
                print("\n===================================================================================")
                print(f"\nKernel: {kernel} Result")
                print(f'-----------------------\n')
                print(f"True Negative: {TN}")
                print(f"True Positive: {TP}")
                print(f"False Negative: {FN}")
                print(f"False Positive: {FP}\n")
                print(f"Accuracy (Difference of acutal & predicted values): {report.accuracy()}")
                print(f"Precision (measuring exactness): {report.precision()}")
                print(f"Recall (measuring completeness): {report.recall()}")
                print(f"F1 Score(compromise between Precision and Recall): {report.f1_score()}")
                print("=====================================================================================\n")


    
    
    
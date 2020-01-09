# Welcome to Part 3 - Classification!

Unlike regression where you predict a continuous number, you use classification to predict a category. There is a wide variety of classification applications from medicine to marketing. Classification models include linear models like Logistic Regression, SVM, and nonlinear ones like K-NN, Kernel SVM and Random Forests.

In this part, you will understand and learn how to implement the following Machine Learning Classification models:


1. Logestic Regression 
> Explanation of Logestic Regression [video](https://www.youtube.com/watch?v=7qJ7GksOXoA).

2. K Nearest Neighbors(KNN)
> Explanation of KNN [video](https://www.youtube.com/watch?v=MDniRwXizWo).

3. Support Vector Machine(SVM)
4. SVM Kernels
5. Naive Bayes 
> Explanation of Naive Bayes Classification [video](https://www.youtube.com/watch?v=CPqOCI0ahss).
6. Decision Tree Classification
> Explanation of Decision Tree Classification [video](https://www.youtube.com/watch?v=DCZ3tsQIoGU).
7. Random Forest Classification
> Explanation of Random Forest Classification [video](https://www.youtube.com/watch?v=D_2LkhMJcfY).

In this Part 3 you learned about 7 classification models. Like for Part 2 - Regression, that's quite a lot so you might be asking yourself the same questions as before:

    What are the pros and cons of each model ?
    How do I know which model to choose for my problem ?
    How can I improve each of these models ?

Again, let's answer each of these questions one by one:

1. What are the pros and cons of each model ?

Please find [here](https://github.com/mHamzaHanif/machine_learning/blob/dev/Part%203:%20Classification/Classification-Pros-Cons.pdf) a cheat-sheet that gives you all the pros and the cons of each classification model.

2. How do I know which model to choose for my problem ?

Same as for regression models, you first need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Logistic Regression or SVM.

If your problem is non linear, you should go for K-NN, Naive Bayes, Decision Tree or Random Forest.

Then which one should you choose in each case ? You will learn that in Part 10 - Model Selection with k-Fold Cross Validation.

Then from a business point of view, you would rather use:

- Logistic Regression or Naive Bayes when you want to rank your predictions by their probability. For example if you want to rank your customers from the highest probability that they buy a certain product, to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your problem is non linear.

- SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments, for example some market segments you identified earlier with clustering.

- Decision Tree when you want to have clear interpretation of your model results,

- Random Forest when you are just looking for high performance with less need for interpretation. 

3. How can I improve each of these models ?

Same answer as in Part 2: 

In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

    the parameters that are learnt, for example the coefficients in Linear Regression,
    the hyperparameters.

The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about. So for those of you already interested in improving your model performance and doing some parameter tuning, feel free to jump directly to Part 10 - Model Selection.

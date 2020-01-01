# Welcome to Part 2 - Regression!


Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If your independent variable is time, then you are forecasting future values, otherwise your model is predicting present but unknown values. Regression technique vary from Linear Regression to SVR and Random Forests Regression.

In this part, you will understand and learn how to implement the following Machine Learning Regression models:

1. Simple Linear Regression
> Explanation Simple Linear Regression [video](https://www.youtube.com/watch?v=CtKeHnfK5uA)
2. Multiple Linear Regression
3. Polynomial Linear Regression
4. Support Vector Regression(SVR) 
> Explanation of SVR [video](https://www.youtube.com/watch?v=Y6RRHw9uN9o)
5. Decision Tree Regression
> Explanation of Decision Tree Regression [video](https://www.youtube.com/watch?v=DCZ3tsQIoGU)
6. Random Forest Regression
> Explanation of Random Forest Regression [video](https://www.youtube.com/watch?v=D_2LkhMJcfY
)

After learning about these six regression models, you are probably asking yourself the following questions:

    What are the pros and cons of each model?

    How do I know which model to choose for my problem?

    How can I improve each of these models?

Let's answer each of these questions one by one:

    1. What are the pros and cons of each model ?

Please find [here](https://github.com/mHamzaHanif/machine_learning/blob/02_regression/Part%202:%20Regression/materail-for-README/pros_cons_regression.pdf) a cheat-sheet that gives you all the pros and the cons of each regression model.

    2. How do I know which model to choose for my problem ?

First, you need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Simple Linear Regression if you only have one feature, and Multiple Linear Regression if you have several features.

If your problem is non linear, you should go for Polynomial Regression, SVR, Decision Tree or Random Forest. Then which one should you choose among these four ? That you will learn in Part 10 - Model Selection. The method consists of using a very relevant technique that evaluates your models performance, called k-Fold Cross Validation, and then picking the model that shows the best results. Feel free to jump directly to Part 10 if you already want to learn how to do that.

    3. How can I improve each of these models ?

In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

    1. the parameters that are learnt, for example the coefficients in Linear Regression,

    2. the hyperparameters.

The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about. So for those of you already interested in improving your model performance and doing some parameter tuning, feel free to jump directly to Part 10 - Model Selection.

And as a BONUS, please find here some [slides](https://github.com/mHamzaHanif/machine_learning/blob/02_regression/Part%202:%20Regression/materail-for-README/Regularization.pdf) we made about Regularization.

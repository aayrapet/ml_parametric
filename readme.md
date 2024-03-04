In this project I aim to develop using only math and numpy existing parametric models such as Linear Regression, Logistic Regression, Neural Networks (incoming). The project is organised in modules:

-base -> module that stores Gradient Descent optimiser which is a classic Gradient Descent for Generalised Linear Models It has extensions such as stochastic gradient descent and newton raphson. Moreover, this module stores BaseEstimator which is the base estimator for GLM such as Linear Regression and Logistic Regression

-linear -> module that stores Linear Regression model

-logistic -> module that stores Logistic Regression model

-metrics -> module that stores metrics used for

regression:

-mse 
-mae

classification:

-accuracy
-recall
-precision
-f1 

others:

confusion matrix
Cross Validation 

-err_handl -> module that does error management in the modules

-dgp -> module that generates data with different characteristics used for linear regression

-data -> example data for linear and logistic regressions

All the work is tested and summarised in test notebook where these both models are excecuted, tested and evaluated
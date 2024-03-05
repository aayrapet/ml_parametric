# Machine Learning Project

In this project, the aim is to develop parametric models using only math and NumPy. The project is organized into the following modules:

## Modules

- **base**: Contains the Gradient Descent optimizer, which includes classic Gradient Descent for Generalized Linear Models. It also has extensions such as stochastic gradient descent and Newton-Raphson. Additionally, this module stores `BaseEstimator`, which serves as the base estimator for GLM, such as Linear Regression and Logistic Regression.

- **linear**: Stores the Linear Regression model.

- **logistic**: Stores the Logistic Regression model.

- **autoselect**: Stores the Autoselection of variables class. It selects variables using information Criteria defined by user. It has three oprions: backward, forward and stepwise regressions

- **metrics**: Stores various metrics used for:

  - Regression:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
  - Classification:
    - Accuracy
    - Recall
    - Precision
    - F1 Score
  - Others:
    - Confusion Matrix
    - Cross Validation

- **err_handl**: Manages errors within the modules.

- **dgp**: Generates data with different characteristics used for linear regression.

- **data**: Example data for linear and logistic regressions.

## Testing

All the work is thoroughly tested and summarized in the test notebook. This notebook executes, tests, and evaluates both models.

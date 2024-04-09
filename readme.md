# Machine Learning Project

In this project, the aim is to develop parametric models using only math and NumPy. The project is organized into the following modules:


    

## Modules

- **base**: Contains the `GradientDescent` optimizer, which includes classic Gradient Descent for Generalized Linear Models. It also has extensions such as stochastic gradient descent, Newton-Raphson and coordinate descent for L1,L2 and L3 regularisations. Additionally, this module stores `BaseEstimator`, which serves as the base estimator for GLM, such as Linear Regression and Logistic Regression.

- **linear**: Stores the `LinearRegression` model.  

- **logistic**: Stores the `LogisticRegression` model:
  - Binary classification 
  - Multiclass classification
    
    Additionnaly, inference statistics are made on both of them using inverse of block Hessian matrix

- **autoselect**: Stores the `Autoselection` of variables class. It selects variables using information Criteria defined by user. It has three options: backward, forward and stepwise regressions

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

- **dgp**: Generates data with different characteristics used for linear regression and logistic regression



## Testing

All the work is thoroughly tested and summarized in the notebook folder. These notebooks execute, test, and evaluate GLM models.





## Versions

- Python Version: 3.11.4
- Numpy Version : 1.26.4
- Pandas Version : 2.2.1

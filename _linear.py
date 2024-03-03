"""
Linear Regression
"""

# Author: Artur Ayrapetyan

import pandas as pd
import _err_handl as erh
import numpy as np
from typing import Literal

from _base import BaseEstimator
from scipy.stats import t


class LinearRegression(BaseEstimator):
    """
    Initialize a Linear Regression model.

    Parameters
    ----------
    solver : Literal["sdg", "gd", "nr", "ols"]
        The solver algorithm to use for optimization.
        - 'sdg' : Stochastic Gradient Descent.
        - 'gd' : Gradient Descent.
        - 'nr' : Newton-Raphson.
        - 'ols' : Ordinary Least Squares.

    add_intercept : bool, optional, default=True
        Whether to add an intercept term to the model. If True, a constant
        (a.k.a. bias or intercept) will be added to the decision function.

    learning_rate : float, optional, default=0.001
        The learning rate for the optimization algorithms (if applicable).
        Only used for 'sdg' and 'gd' solvers.

    tol_level : float, optional, default=1e-06
        Tolerance level for convergence criteria. The optimization
        algorithm stops when the change in the cost function is less than
        this value.Not used for ols solver.

    max_iteration : int, optional, default=100
        The maximum number of iterations to perform during optimization.
        Not used for ols solver.

    mini_batch_size : int, optional, default=32
        The size of mini-batches used in Stochastic Gradient Descent.
        Ignored for other solvers.

    """

    def __init__(
        self,
        solver: Literal["sgd", "gd", "nr", "ols"],
        add_intercept: bool = True,
        learning_rate: float = 0.001,
        tol_level: float = 1e-06,
        max_iteration: int = 100,
        mini_batch_size: int = 32,
        need_to_store_results: bool = True,
    ):

        # only these solvers are allowed
        if solver not in ["sgd", "gd", "nr", "ols"]:
            raise ValueError("no known algorithm  provided")
        # make sure that good data types are used
        erh.check_arguments_data(
            (solver, str),
            (add_intercept, bool),
            (learning_rate, float),
            (tol_level, float),
            (max_iteration, int),
            (mini_batch_size, int),
            (need_to_store_results, bool),
        )

        super().__init__(
            solver,
            add_intercept,
            learning_rate,
            tol_level,
            max_iteration,
            mini_batch_size,
            need_to_store_results,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        The central function that calculates estimator of Linear Regression based on Gradient descent algorithm (or OLS)

        Parameters
        ----
        x : np.ndarray -> matrix of x variables
        y : np.ndarray -> vector of y

        Returns
        ----
        Returns estimator "B" of Linear Regression problem

        Example
        -----

        #run Linear Regression
        >>> model=LinearRegression(solver="ols")
        >>> model.fit(x_train,y_train)

        array([ -1.02713264,   5.24093556,   1.51606519,  -7.005])

        """
        erh.check_arguments_data((x, np.ndarray), (y, np.ndarray))
        result_param = super().fit_base(x, y, "linear")

        return result_param

    def predict(
        self, x: np.ndarray, param_if_not_kept: np.ndarray = None
    ) -> np.ndarray:
        """
        Predict y using x matrix

        Parameters
        ----
        x : np.ndarray -> matrix of x variables used for prediction

        Returns
        ----
        Returns vector of y predicted as an array

        Example
        -----

        #run Linear Regression
        >>> model=LinearRegression(solver="ols")
        >>> model.fit(x_train,y_train)
        >>> y_test_predicted=model.predict(x_test)
        >>> y_test_predicted
        array([ -1.02713264,   5.24093556,   1.51606519,  -7.005])

        """
        # make sure we import numpy matrix
        erh.check_arguments_data((x, np.ndarray))
        pred = super().predict_linear(x, param_if_not_kept)

        return pred

    def get_inference(self, biased: bool = True) -> pd.DataFrame:
        """
        Function to calculate variance covariance matrix, significance levels for parameters, Information Criteria (IC)

        Parameters
        ---------
        biased: bool -> Divide Sum of squared residuals by N (biased) or (N-p) (unbiased)
            where N is number of rows of X, p is number of columns of X

        Returns
        ------
        variance covariance matrix available in self.vcov_matrix,
        significance levels for parameter savailable as return statement,
        Information Criteria available in self.criteria
        There are two IC: AIC and BIC (based on LogLikelihood (with suffix _ll) and errors( with suffix _err))

        AIC_ll = -2 * LL + 2 * (p)

        BIC_ll = -2 * LL + np.log(N) * (p)

        BIC_err = N * np.log(variance) + p * np.log(N)

        AIC_err = N * np.log(variance) + 2 * p + N + 2

        With LL a LogLikelihood of normal distribution :

        LL = -0.5 * N * np.log(2 * np.pi * variance) - (SCR) / (2 * variance)

        and variance = SSR / N

        Examples
        --------

        #run Linear Regression
        >>> model=LinearRegression(solver="ols")
        >>> model.fit(x,y)
        >>> model.get_inference()
        params   std    t value  p value
        0  -0.1  0.01   -0.2    0.76
        1   2.5   0.3    534    0.0
        2   1.2   2      29     0.0


        """
        erh.check_arguments_data((biased, bool))
        # better to pass predictions to attributes
        SCR = np.sum((self.y - (self.x @ self.params)) ** 2, axis=0)
        N = self.x.shape[0]
        p = self.x.shape[1]
        variance = SCR / N if biased else SCR / (N - p)
        self.variance_residuals = variance

        LL = -0.5 * N * np.log(2 * np.pi * variance) - (SCR) / (2 * variance)
        AIC_ll = -2 * LL + 2 * (p)
        BIC_ll = -2 * LL + np.log(N) * (p)
        BIC_err = N * np.log(variance) + p * np.log(N)
        AIC_err = N * np.log(variance) + 2 * p + N + 2

        criteria = {}
        criteria["LL"] = LL
        criteria["AIC_ll"] = AIC_ll
        criteria["BIC_ll"] = BIC_ll
        criteria["BIC_err"] = BIC_err
        criteria["AIC_err"] = AIC_err

        self.criteria = criteria
        # calculate variance covariance matrix and p values of parameters
        vcov_matrix = variance * np.linalg.inv(self.x.T @ self.x)
        self.vcov_matrix = vcov_matrix
        std_params = np.sqrt(np.diagonal(vcov_matrix))
        t_value = self.params / std_params
        p_value = [(2 * t.sf(np.abs(el), df=N - p)) for el in t_value]
        result = pd.DataFrame(
            np.column_stack((self.params, std_params, t_value, np.round(p_value, 4))),
            columns=["params", "std", "t value", "p value"],
        )
        return result

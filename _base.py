"""
Generalized Linear Models
"""

# Author: Artur Ayrapetyan

from _autoselect import AutoSelect
import pandas as pd
import _err_handl as erh
import numpy as np
from typing import Literal
import warnings
import math as mt
from itertools import product


class GradientDescent:
    """
    A class implementing gradient descent optimization algorithms for various solvers.

    Parameters
    ----------
    regression : "linear"|"logistic"|"logistic softmax"
        Type of regression to perform on data

    solver : str
        The optimization algorithm to use. It can be 'gd' for gradient descent,
        'sgd' for stochastic gradient descent, 'nr' for Newton-Raphson, or 'ols' for
        Ordinary Least Squares.

    learning_rate : float
        The learning rate used in the optimization algorithms. Higher learning rates
        can lead to faster convergence but may also cause instability.

    tol_level : float
        The tolerance level for convergence criteria. The optimization algorithm
        stops when the change in the cost function is less than this value.

    max_iteration : int
        The maximum number of iterations to perform during optimization.

    mini_batch_size : int
        The size of mini-batches used in stochastic gradient descent. Ignored for
        other solvers.

    x : array_like
        The feature matrix.

    y : array_like
        The target vector.

    Attributes
    ----------
    solver : str
        The optimization algorithm being used.

    learning_rate : float
        The learning rate for the optimization algorithms.

    tol_level : float
        Tolerance level for convergence criteria.

    x : array_like
        The feature matrix.

    y : array_like
        The target vector.

    mini_batch_size : int
        The size of mini-batches used in stochastic gradient descent.

    newton_raphson : bool
        Indicates if the Newton-Raphson algorithm is being used.

    change : float
        The change in parameter values between iterations.

    max_iteration : int
        The maximum number of iterations to perform during optimization.
    print_message : bool
        Print or not the message of convergence (recommended: True)
    """

    def __init__(
        self,
        regression: Literal["linear", "logistic"],
        solver,
        learning_rate,
        tol_level,
        max_iteration,
        mini_batch_size,
        x,
        y,
        print_message: bool = True,
        alpha: float=None,
        lambd : float =None
    ):

        self.regression = regression
        self.solver = solver
        self.learning_rate = learning_rate
        self.tol_level = tol_level
        self.x = x
        self.y = y
        self.mini_batch_size = mini_batch_size
        self.newton_raphson = False
        self.change = float("inf")
        self.max_iteration = max_iteration
        self.print_message = print_message
        self.alpha=alpha
        self.lamdb=lambd

    def optimiser_update_parameter(self, B, x, y):
        """
        Update the parameter B based on the optimization algorithm.

        This method updates the parameter B based on the optimization algorithm being used,
        such as gradient descent or Newton-Raphson.

        Parameters
        ----------
        B : array_like
            The parameter vector to be updated.

        x : array_like
            The feature matrix.

        y : array_like
            The target vector.

        Returns
        -------
        array_like
            The updated parameter vector.

        """
        

        if self.regression == "linear":
            prediction = x @ B
            gradient = (2 * (y - prediction)).T @ x

        elif self.regression == "logistic":

            linear_predictions = x @ B
            #modif1
            # numerator = np.exp(linear_predictions)

            # denominator = np.zeros(linear_predictions.shape[0])
            # nb_cols = (
            #     linear_predictions.shape[1]
            #     if linear_predictions.ndim > 1
            #     else linear_predictions.ndim
            # )

            # for i in range(nb_cols):
            #     denominator = (
            #         denominator + numerator[:, i]
            #         if linear_predictions.ndim > 1
            #         else denominator + numerator
            #     )
            # denominator = denominator + np.exp(0)  # we add reference class
            # denominator = (
            #     np.column_stack([denominator] * linear_predictions.shape[1])
            #     if linear_predictions.ndim > 1
            #     else denominator
            # )

            # proba = (
            #     numerator / denominator
            # )
            
            
            #SOFTMAX FUNCTION to calculate probability 
            #here i simplify the code with numpy functions
            P=np.exp(linear_predictions)
            
            norms = np.diag(1/(P.sum(1)+1))
            
            proba = norms@P
            
            gradient = x.T @ (y - proba)
            
            #adjust gradient
            ##MODIFIMPORTANT1
            
            gradient=gradient.T.reshape((gradient.shape[0]*gradient.shape[1],1))
           
        
           
        # if self.newton_raphson and self.regression == "logistic softmax":
        #     raise ValueError("not done yet,incoming")

        
        #MODIFIMPORTANT2
        if self.regression=="logistic":
            B=B.T.reshape((B.shape[0]*B.shape[1],1))
        # If learn_rate is an scalar (GD)
        if not self.newton_raphson:
            
            B_new = B + self.learning_rate * gradient

        # If learn_rate is an inverse of hessian (NR)
        elif self.newton_raphson:
            if self.regression == "logistic":

                nb_dimensions = 1 if y.ndim == 1 else y.shape[1]
                
                xx = block_diagonal(x, nb_dimensions)
                if proba.ndim == 1:
                    proba = proba.reshape(proba.shape[0], 1)
                ww = block_prob_matrix(proba, nb_dimensions)

                self.learning_rate = np.linalg.inv(xx.T @ ww @ xx)
           
            
          
                        
            B_new = B + self.learning_rate @ gradient

        change = np.sum((B - B_new) ** 2)
        B = B_new.copy()
        self.change = change

        return B_new

    def optimiser_verify_condition(self, iteration, local_max_iter):
        """
        Verify convergence condition for the optimization algorithm.

        This method checks if the optimization algorithm has converged within the specified
        maximum number of iterations or if it has reached the convergence tolerance level.

        Parameters
        ----------
        iteration : int
            The current iteration number of the optimization algorithm.

        local_max_iter : int
            The maximum number of iterations specified for the optimization algorithm.

        Notes
        -----
        - If the algorithm converges within the specified maximum iterations or reaches
          the convergence tolerance level, the optimization loop is stopped.
        - If the algorithm does not converge within the maximum iterations and does not
          meet the convergence criteria, a message indicating bias in the calculated
          parameters is displayed, but still calculates the parameter.

        """
        message_good = f"algorithm did  converge under {local_max_iter} iterations (at {iteration} iterations)"
        message_bad = f"algorithm did not converge under {local_max_iter} iterations,so the calculated parameter is biased"
        # during all iterations
        if iteration < local_max_iter:
            if self.change < self.tol_level:
                self.stop_loop = True

                print(message_good) if self.print_message else None
        # on the final iteration
        elif iteration == local_max_iter:
            if self.change < self.tol_level:
                self.stop_loop = True
                print(message_good) if self.print_message else None
            if self.change > self.tol_level:
                print(message_bad) if self.print_message else None

    def optimiser_algorithm_classic(self):
        """
        Perform classic gradient descent optimization based on the selected solver.

        This method applies various gradient descent optimization algorithms such as
        Newton-Raphson, batch gradient descent, stochastic gradient descent, or
        ordinary least squares based on the specified solver.

        Returns
        -------
        array_like
            If the solver is 'ols' (Ordinary Least Squares), the coefficients (B)
            of the linear regression model are returned.

        Notes
        -----
        - For the 'nr' (Newton-Raphson) solver, the learning rate is calculated as
          the inverse of the Hessian matrix of the feature matrix (X).
        - For the 'gd' (batch gradient descent) solver, vanilla gradient descent is
          applied to update the parameter iteratively until convergence or reaching
          the maximum number of iterations.
        - For the 'sgd' (stochastic gradient descent) solver, stochastic gradient
          descent or mini-batch gradient descent is applied. Mini-batches are randomly
          sampled from the dataset for each iteration.
        - For the 'ols' (Ordinary Least Squares) solver, the closed-form solution
          is calculated directly.



        """

        self.stop_loop = False
        if self.lamdb!=None or self.alpha!=None:
            pass
        elif self.solver == "nr":
            self.newton_raphson = True
            if self.regression == "linear":
                self.learning_rate = np.linalg.inv(2 * self.x.T @ self.x)
            stochastic = False
        elif self.solver == "gd":
            stochastic = False
        elif self.solver == "sgd":
            stochastic = True
        elif self.solver == "ols":
            B = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y
            return B
        p = self.x.shape[1]
        if self.regression == "logistic":
            
           
            if self.y.ndim == 1:
                
                B = np.zeros((self.x.shape[1],1)) #modif2
                self.y=self.y.reshape((self.y.shape[0],1))#modif3
            else:
                B = np.zeros((self.x.shape[1], self.y.shape[1]))
         
        elif self.regression == "linear":
            B = np.zeros((self.x.shape[1]))
        with warnings.catch_warnings(record=True) as w:
            #Coordinate descent : attention, used only for linear regression        
            if self.lamdb!=None or self.alpha!=None:
                self.alpha = self.alpha if self.alpha != None else 0
                self.lamdb = self.lamdb if self.lamdb != None else 0
                
                for iteration in range(1,self.max_iteration+1):
                     B_old=B.copy()
                     for j in range(self.x.shape[1]):
                                               
                         pred_y=self.x@B
                   
                         gradient=(self.x[:,j].T)@(self.y-pred_y)

                         B[j]=B[j]+self.learning_rate*gradient  #0.001 as learn rate
                         if j==0:
                             pass
                         else:
                            B[j]=np.sign(B[j])*max(np.abs(B[j])-self.lamdb*self.alpha,0)/(1 + (self.lamdb * (1 - self.alpha))); 
                        
                         
                         
                     self.change  = np.sum((B_old - B)**2)  
                     self.optimiser_verify_condition(
                            iteration, self.max_iteration
                        )
                        
                     if self.stop_loop:
                            break
                
            # Gradient descent/newton Raphson (or batch gradient descent)
            elif not stochastic:
                local_max_iter = self.max_iteration
                ft=True
                for i in range(1, self.max_iteration + 1):
                    #MODIFIMPORTANT3
                    if self.regression=="logistic":
                      if ft:
                          ft=False
                      else: 
                          B=np.reshape(B, (-1, (self.y.shape[1])), order='F')
                        
                    B = self.optimiser_update_parameter(B, self.x, self.y)
                    self.optimiser_verify_condition(i, local_max_iter)
                    if self.stop_loop:
                        break
            # Stochastic gradient descent (or mini-batch gradient descent)
            elif stochastic:
                local_max_iter = self.max_iteration
                iteration = 1
                N = self.x.shape[0]
                number = mt.ceil(N / self.mini_batch_size)
                ft=True
                for j in range(1, self.max_iteration + 1):
                    indices = np.random.permutation(N)
                    x = self.x[indices]
                    y = self.y[indices]
                    # Iterate over mini-batches
                    for i in range(0, N, self.mini_batch_size):
                        x_batch = x[i : i + self.mini_batch_size]
                        y_batch = y[i : i + self.mini_batch_size]
                        if self.regression=="logistic":
                          if ft:
                              ft=False
                          else: 
                              B=np.reshape(B, (-1, (self.y.shape[1])), order='F')
                        B = self.optimiser_update_parameter(B, x_batch, y_batch)
                        self.optimiser_verify_condition(
                            iteration, local_max_iter * number
                        )
                        iteration = iteration + 1
                        if self.stop_loop:
                            break
                    if self.stop_loop:
                        break
           
            
            # if overflow  is present in warnings (vexploding effect of gradient descent due to multicollinearity, non normalised data, learning rate)

        for warning in w:
            if "overflow" in str(warning.message).lower():
                B = None
                raise ValueError(
                    "Attention, Gradient descent overlow due to potential collinearity "
                    + "or not normalised data. "
                    + "\n"
                    + "Please select another solver/learning rate "
                    + "or correct data"
                )

        return B


##############################################################
#


class BaseEstimator:
    """
    Initialize a Regression model.

    Parameters
    ----------
    regression :  Literal["linear", "logistic", "logistic softmax"]
        -linear : Perform Linear regression
        -logistic : Perform binary/OVR logistic regression using sigmoid function
        -logistic softmax: Perform multinomial logistic regression using softmax function
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
        solver: str,
        add_intercept: bool = True,
        learning_rate: float = 0.001,
        tol_level: float = 1e-06,
        max_iteration: int = 100,
        mini_batch_size: int = 32,
        need_to_store_results: bool = True,
        print_message: bool = True,
    ):

        # define mandatory field:
        self.solver = solver
        # define default fields:
        self.add_intercept = add_intercept
        self.learning_rate = learning_rate
        self.tol_level = tol_level
        self.max_iteration = max_iteration
        self.mini_batch_size = mini_batch_size
        self.need_to_store_results = need_to_store_results
        self.print_message = print_message
        # define  fields that will be calculated after:
        self.predictions = "not calculated yet"
        self.params = "not calculated yet"
        self.variance_residuals = "not calculated yet"
        self.vcov_matrix = "not calculated yet"
        self.criteria = "not calculated yet"
        self.x = None
        self.y = None

    def add_intercept_f(self, x: np.ndarray) -> np.ndarray:
        """
        Simple function to add (or not) an intercept to X matrix
        """
        if self.add_intercept:
            x = np.column_stack((np.ones(x.shape[0]), x))
        return x

    def fit_base(
        self, x: np.ndarray, y: np.ndarray, methodic: Literal["linear", "logistic"],
        alpha: float=None, lambd: float=None
    ) -> np.ndarray:
        """
        The central function that calculates estimator of  Regression based on Gradient descent algorithm (or OLS)

        Parameters
        ----
        x : np.ndarray -> matrix of x variables
        y : np.ndarray -> vector of y

        Returns
        ----
        Returns estimator "B" of  Regression problem

        Example
        -----


        """

        # make sure we import numpy matrix
        erh.check_arguments_data((methodic, str))
        # add intercept (or not)
        x = self.add_intercept_f(x)
        alg = GradientDescent(
            methodic,
            self.solver,
            self.learning_rate,
            self.tol_level,
            self.max_iteration,
            self.mini_batch_size,
            x,
            y,
            self.print_message,
            alpha,
            lambd
        )
        result_param = alg.optimiser_algorithm_classic()
        
        if methodic=="logistic":
           if y.ndim==1: 
               nb_classes_1=1
           else:
               nb_classes_1=y.shape[1]
           result_param=np.reshape(result_param, (-1, (nb_classes_1)), order='F')
        
        if self.need_to_store_results:
            self.params = result_param
            self.x = x
            self.y = y

        return result_param
    
    
    
    
    
    
    def predict_linear(
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


        """
        if self.need_to_store_results:
            parameter = self.params
            if param_if_not_kept is not None:
                print("introduced param ignored, as we store results")
        else:
            parameter = param_if_not_kept
            if param_if_not_kept is None:
                raise ValueError(
                    "we dont store results, so you neeed to introduce param"
                )

        # add intercept (or not)
        x = self.add_intercept_f(x)

        pred = x @ parameter

        return pred

    def autoselection(
        self,
        method: Literal["backward", "forward", "stepwise"],
        criterion: Literal[
            "BIC_ll",
            "AIC_ll",
            "AIC_err",
            "BIC_err",
            "LL",
        ] = "BIC_ll",
        print_message: bool = True,
    ) -> np.ndarray:
        """
        Perform automatic variable selection for the model.

        Parameters:
            method (str): The method to use for variable selection. It can be one of: "backward", "forward", or "stepwise".
            criterion (str, optional): The criterion to use for variable selection.
                Options include: "BIC_ll", "AIC_ll", "AIC_err", "BIC_err", or "LL".
                Defaults to "BIC_ll".

        Returns:
            np.ndarray: An array containing the indices of the selected variables.

        Raises:
            ValueError: If the model has not been fitted (i.e., if `x` or `y` is None).

        Note:
            This method assumes that the model has already been fitted. It uses the provided method
            and criterion to perform automatic variable selection and returns the indices of the selected variables.
        """
        self.print_message = print_message
        if self.x is None or self.y is None:
            raise ValueError("fit the model first, then do autoselect")
        if self.add_intercept:
            x = self.x[:, 1:]
        else:
            x = self.x

        sel = AutoSelect(self, method, criterion)
        index_selected_variables = sel.fit(x, self.y)
        # get back print option
        self.print_message = True
        return index_selected_variables


def block_diagonal(matrix, nb_blocks):
    """
    Duplicates matrix X n times on  a diagonal to get a diagonal block matrix, other blocks being 0

    """
    N, p = matrix.shape
    block = np.zeros((N * nb_blocks, p * nb_blocks))

    for i in range(nb_blocks):
        block[i * N : N * (i + 1), i * p : p * (i + 1)] = matrix
    return block


def block_prob_matrix(matrix, nb_blocks):
    """
    Get block matrix of probabilities.
    On diagonal terms we get diagonal matrices Pki*(1-Pki) .... P(k-1)i*(1-P(k-1)i)
    On other terms we get diagonal matrices Pki*Pli .... P(k-1)i*(P(l-1)i)

    Returns
    ------

    Probability matrix W

    References
    --------

    To visualise go to : https://github.com/aayrapet/ml_parametric/issues/1


    """

    size_block = matrix.shape[0]

    list_values = list(range(nb_blocks))
    all_combinations = list(product(list_values, repeat=2))

    nb = len(list_values)

    W = np.zeros((nb * size_block, nb * size_block))
    i = -1

    el_old = [100]
    for el in all_combinations:

        if el_old[0] == el[0]:
            j = j + 1
        else:
            # go on the next line
            j = 0
            i = i + 1

        # block is respective probabilities matrix that will be injected as diagonal matrix
        if el[0] == el[1]:
            block = matrix[:, el[0]] * (1 - matrix[:, el[0]])
        else:
            block = (-1) * matrix[:, el[0]] * (matrix[:, el[1]])

        W[
            el[0] * size_block : size_block * (i + 1),
            el[1] * size_block : size_block * (j + 1),
        ] = np.diag(block)
        el_old = el

    return W

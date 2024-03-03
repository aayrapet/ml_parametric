"""
Logistic Regression
"""

# Author: Artur Ayrapetyan

import pandas as pd
import _err_handl as erh
import numpy as np
from typing import Literal
from _base import BaseEstimator
from scipy.stats import t


class LogisticRegression(BaseEstimator):
    def __init__(
        self,
        solver: Literal["sgd", "gd", "nr"],
        add_intercept: bool = True,
        learning_rate: float = 0.001,
        tol_level: float = 1e-06,
        max_iteration: int = 100,
        mini_batch_size: int = 32,
        need_to_store_results: bool = True,
        multiclass: Literal["binary", "ovr", "softmax"] = "binary",
    ):

        if solver not in ["sgd", "gd", "nr"]:
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
            (multiclass, str),
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
        self.multiclass = multiclass

        self.proba = "not calculated yet"
        self.true_labels_matrix = "not calculated yet"
        self.params = "not calculated yet"

    def transform_y_vector_to_matrix(self, y: np.ndarray) -> np.ndarray:
        """
        Transforms a target vector into a matrix based on multiclass strategy.

        Parameters:
            y (array-like): The target vector to be transformed.

        Returns:
            array-like: The transformed target matrix.

        Notes:
            If the multiclass strategy is 'ovr' (One-vs-Rest) or 'softmax', the function
            formats the target vector into a matrix using one-hot encoding. It checks
            if the input vector is already in matrix form, and if not, converts it.
        """
        if self.multiclass == "ovr" or self.multiclass == "softmax":
            # if a multiclass vector is introducred then format it as matrix (one hot encoding)
            if len(y.shape) == 1:

                unique_classes = sorted(np.unique(y))
                ft = True
                for unique_class in unique_classes:
                    class_y = np.array([1 if el == unique_class else 0 for el in y])

                    if ft:
                        ft = False

                        result_true_label = class_y
                    else:

                        result_true_label = np.column_stack(
                            (result_true_label, class_y)
                        )

                y = result_true_label

        return y

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

        y = self.transform_y_vector_to_matrix(y)

        if self.multiclass == "binary":
            result_param = super().fit_base(x, y, "logistic")

        elif self.multiclass == "softmax":
            result_param = super().fit_base(x, y, "logistic softmax")
        elif self.multiclass == "ovr":
            ft = True
            for class_y in y.T:

                param = super().fit_base(x, class_y, "logistic")
                if ft:
                    ft = False
                    result_param = param

                else:
                    result_param = np.column_stack((result_param, param))

        if self.need_to_store_results:
            self.params = result_param
            self.true_labels_matrix = (
                y if self.multiclass == "ovr" or self.multiclass == "softmax" else None
            )

        return result_param

    @staticmethod
    def normalise_predictions(proba: np.ndarray) -> np.ndarray:
        """ "
        In case of OVR regression, normalise probabilities
        """
        sum_columns = np.array(([sum(x) for x in zip(*proba.T)]))

        sum_columns = np.column_stack([sum_columns] * proba.shape[1])
        return sum_columns

    @staticmethod
    def from_proba_to_prediction(proba: np.ndarray) -> np.ndarray:
        """
        Converts probability scores to class predictions.

        Parameters:
            proba (np.ndarray): The input array containing probability scores for each class.

        Returns:
            np.ndarray: The array of class predictions converted from probability scores.

        Notes:
            This function takes an array of probability scores and converts them into class
            predictions using the maximum probability criterion. Each row of the input array
            represents the probabilities for different classes, and the function assigns the
            class with the highest probability as the predicted class for each sample.
        """

        result_matrix = np.zeros((proba.shape[0], proba.shape[1]))
        for i, row in enumerate(proba):
            max_index = np.argmax(row)
            new_row = np.zeros(proba.shape[1])
            new_row[max_index] = 1
            result_matrix[i] = new_row

        return result_matrix

    def predict(
        self,
        x: np.ndarray,
        param_if_not_kept: np.ndarray = None,
        threshold: float = 0.5,
    ):
        """
        Predicts class labels for input data.

        Parameters:
            x (np.ndarray): The input data for prediction.
            param_if_not_kept (np.ndarray, optional): Additional parameters for prediction if not kept. Defaults to None.
            threshold (float, optional): Threshold for binary classification. Defaults to 0.5.

        Returns:
            np.ndarray: The array of predicted class labels.

        Notes:
            This function predicts class labels based on the input data and the model's parameters.
            For binary classification, it uses the sigmoid function with a given threshold.
            For multiclass classification with One-vs-Rest (ovr) or Softmax strategy, it calculates
            probabilities using appropriate methods and then converts them to class predictions.
            If predictions need to be stored, it stores the predicted probabilities.

        """

        erh.check_arguments_data((x, np.ndarray), (threshold, float))
        linear_predictions = super().predict_linear(x, param_if_not_kept)

        if self.multiclass == "binary":
            # use sigmoid function
            proba = 1 / (1 + np.exp(-(linear_predictions)))
            prediction = np.array([1 if el >= threshold else 0 for el in proba])

        elif self.multiclass == "ovr":
            # use sigmoid function
            proba = 1 / (1 + np.exp(-(linear_predictions)))
            # normalise proba so that they sum up to 1
            normalised_proba = proba / self.normalise_predictions(proba)
            # transform matrix of proba to matrix of prediction
            prediction = self.from_proba_to_prediction(normalised_proba)
        elif self.multiclass == "softmax":

            numerator = np.exp(linear_predictions)

            denominator = np.zeros(linear_predictions.shape[0])
            for i in range(linear_predictions.shape[1]):
                denominator = denominator + numerator[:, i]
            denominator = np.column_stack([denominator] * linear_predictions.shape[1])
            proba = numerator / denominator
            # transform matrix of proba to matrix of prediction
            prediction = self.from_proba_to_prediction(proba)

        if self.need_to_store_results:
            self.proba = proba

        return prediction

    def get_inference(self):
        """
        Calculates inference statistics and parameter estimates.

        Returns:
            DataFrame: A DataFrame containing parameter estimates, standard deviations, t-values, and p-values.

        Raises:
            ValueError: If probabilities are not computed (run .predict() first).

        Notes:
            This function calculates inference statistics and parameter estimates based on the
            stored probabilities and input data. It computes likelihood, AIC (Akaike Information Criterion),
            and BIC (Bayesian Information Criterion) values. Then, it calculates t-values and p-values
            for each parameter. The results are returned as a DataFrame.
        """
        if self.multiclass in ["ovr", "softmax"]:
            raise ValueError("no done yet, incoming")

        if isinstance(self.proba, str):
            raise ValueError("run .predict() first")
        N = self.x.shape[0]
        p = self.x.shape[1]

        w = self.proba * (1 - self.proba)
        Wdiag = np.diag(w)
        std_params = np.sqrt(np.diag(np.linalg.inv(self.x.T @ Wdiag @ self.x)))
        LL = self.y.T @ self.x @ self.params - np.sum(
            np.log(1 + np.exp(self.x @ self.params))
        )

        AIC_ll = -2 * LL + 2 * (p)
        BIC_ll = -2 * LL + np.log(N) * (p)

        criteria = {}
        criteria["LL"] = LL
        criteria["AIC_ll"] = AIC_ll
        criteria["BIC_ll"] = BIC_ll

        self.criteria = criteria
        # calculate variance covariance matrix and p values of parameters

        t_value = self.params / std_params
        p_value = [(2 * t.sf(np.abs(el), df=N - p)) for el in t_value]
        result = pd.DataFrame(
            np.column_stack((self.params, std_params, t_value, np.round(p_value, 4))),
            columns=["params", "std", "t value", "p value"],
        )
        return result

"""
Metrics for model performance
"""

# Author: Artur Ayrapetyan


import pandas as pd
import _err_handl as erh
import numpy as np
from _linear import LinearRegression
from _logistic import LogisticRegression
from typing import Type
from typing import List
import types
import math as mt


def accuracy_score(actual, predicted):
    """
    Calculate the accuracy score of predicted classes.

    Parameters:
    actual (array): The actual class labels.
    predicted (array): The predicted class labels.

    Returns:
    float: The accuracy score.


    """
    # check conditions
    erh.check_arguments_data((actual, np.ndarray), (predicted, np.ndarray))
    # if binary classification so one dim vector of labels
    if len(actual.shape) == 1:

        erh.check_equal_size_vectors(actual, predicted)
        erh.check_unique_classes(actual, predicted)
        res = np.sum(actual == predicted) / len(actual)

    # if multiclass classification so  matrix of labels with several vectors
    elif len(actual.shape) > 1:
        # substract one from another and when at least one value is negative -> this is a bad prediction
        dif_matrix = actual - predicted
        sums = 0
        for row in dif_matrix:
            if np.any(row < 0):
                pass
            else:
                sums = sums + 1

        res = sums / dif_matrix.shape[0]

    # calculate
    return res


def recall_score(actual, predicted):
    """
    Calculate the recall score of predicted classes.

    Recall measures the ability of a classifier to find all relevant cases within a dataset.

    Parameters:
    actual (array-like): The actual class labels.
    predicted (array-like): The predicted class labels.

    Returns:
    float: The recall score.


    """
    # check conditions
    erh.check_arguments_data((actual, np.ndarray), (predicted, np.ndarray))
    erh.check_equal_size_vectors(actual, predicted)
    erh.check_unique_classes(actual, predicted)
    erh.check_two_classes(actual)
    # calculate
    True_positives = np.sum((actual == 1) & (predicted == 1))
    False_negatives = np.sum((actual == 1) & (predicted == 0))
    recall = True_positives / (True_positives + False_negatives)
    return recall


def precision_score(actual, predicted):
    """
    Calculate the precision score of predicted classes.

    Precision measures the ability of a classifier not to label as positive a sample that is negative.

    Parameters:
    actual (array-like): The actual class labels.
    predicted (array-like): The predicted class labels.

    Returns:
    float: The precision score.


    """
    # check conditions
    erh.check_arguments_data((actual, np.ndarray), (predicted, np.ndarray))
    erh.check_equal_size_vectors(actual, predicted)
    erh.check_unique_classes(actual, predicted)
    erh.check_two_classes(actual)
    # calculate

    True_positives = np.sum((actual == 1) & (predicted == 1))
    False_negatives = np.sum((actual == 0) & (predicted == 1))
    precision = True_positives / (True_positives + False_negatives)
    return precision


def f1_score(actual, predicted):
    """
    Calculate the F1 score of predicted classes.

    The F1 score is the harmonic mean of precision and recall.

    Parameters:
    actual (array-like): The actual class labels.
    predicted (array-like): The predicted class labels.

    Returns:
    float: The F1 score.


    """
    # check conditions
    erh.check_arguments_data((actual, np.ndarray), (predicted, np.ndarray))
    erh.check_equal_size_vectors(actual, predicted)
    erh.check_unique_classes(actual, predicted)
    erh.check_two_classes(actual)
    # calculate
    recall = recall_score(actual, predicted)
    precision = precision_score(actual, predicted)
    f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score


def confusion_matrix(actual, predicted):
    """
    Calculate the confusion matrix of predicted classes.

    Parameters:
    actual (array-like): The actual class labels.
    predicted (array-like): The predicted class labels.

    Returns:
    pd.DataFrame: The confusion matrix.

    Raises:
    TypeError: If actual or predicted data is not of type np.ndarray.
    ValueError: If the sizes of actual and predicted vectors are not equal,
                or if they don't have the same number of unique classes.
    """
    # check conditions
    erh.check_arguments_data((actual, np.ndarray), (predicted, np.ndarray))
    erh.check_equal_size_vectors(actual, predicted)
    erh.check_unique_classes(actual, predicted)

    # calculate
    nb_classes = len(np.unique(actual))
    unique_class_names = np.unique(actual)
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    for true, pred in zip(actual, predicted):
        confusion_matrix[true, pred] = confusion_matrix[true, pred] + 1

    ConfusionMatrix = pd.DataFrame(
        confusion_matrix,
        index=[["Actual"] * len(unique_class_names), unique_class_names],
        columns=[["Predicted"] * len(unique_class_names), unique_class_names],
    )
    return ConfusionMatrix


def mse_score(
    y: np.ndarray,
    predictions: np.ndarray,
):
    # check conditions
    erh.check_arguments_data((y, np.ndarray), (predictions, np.ndarray))
    erh.check_equal_size_vectors(y, predictions)
    # calculate
    mse = np.sum((y - predictions) ** 2, axis=0) / y.shape[0]
    return mse


def mae_score(
    y: np.ndarray,
    predictions: np.ndarray,
):

    # check conditions
    erh.check_arguments_data((y, np.ndarray), (predictions, np.ndarray))
    erh.check_equal_size_vectors(y, predictions)
    mae = np.sum(np.abs(y - predictions), axis=0) / y.shape[0]
    return mae


def CrossValidation(
    Class_algorithm: Type,
    x: np.ndarray,
    y: np.ndarray,
    metrics_function: types.FunctionType,
    nb_k_fold: int = 4,
) -> List:
    """
     Cross Validation operation for ML algorithms

    Parameters
    --------
        Class_algorithm (class): ML algorithm (__init__)
        x : x Matrix of data
        y: y Vector of data
        metrics_function: function to calculate model performance with a metric
        nb_k_fold: number of folds for CV

    Returns
    --------
        List of calculated metric, we can take the average to evaluate model performance

    Examples
    --------

    >>> #run Linear Regression
    >>> model=LinearRegression()
    >>> mse_list=CrossValidation(Class_algorithm=model,x=x,y=y,metrics_function=mse_score,nb_k_fold=4)
    >>> mse_list
    [0.01240,
    0.013500,
    0.011301,
    0.012454]

    See Also
    --------
    https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85

    """

    # check if arguments are of good type
    erh.check_arguments_data(
        (Class_algorithm, "__class__"),
        (x, np.ndarray),
        (y, np.ndarray),
        (metrics_function, types.FunctionType),
        (nb_k_fold, int),
    )
    if isinstance(Class_algorithm, LinearRegression):
        if metrics_function not in [mse_score, mae_score]:
            raise ValueError(
                "for Linear regression do only following functions: mse_score, mae_score"
            )
    elif isinstance(Class_algorithm, LogisticRegression):
        if metrics_function not in [
            accuracy_score,
            recall_score,
            precision_score,
            f1_score,
        ]:
            raise ValueError(
                "for Logistic regression do only following functions: accuracy_score,recall_score,precision_score,f1_score"
            )

    # from the intoduced class take alll earlier defined attributes
    # BUT change only the last one in order to discard all changes
    # made in this function to the class attributes like predictions, parameters , input data etc
    Class_algorithm.need_to_store_results = False

    N = x.shape[0]
    index_k_fold = mt.floor(N / nb_k_fold)
    # shuffle data
    indices = np.random.permutation(N)
    x = x[indices]
    y = y[indices]

    # Iterate over folds+do modelling
    iteration = 1
    metrics_list = []
    if len(y.shape) == 1:
        concatenate_function = np.hstack
    elif len(y.shape) > 1:
        concatenate_function = np.row_stack

    for i in range(0, N, index_k_fold):

        left_index = i
        right_index = left_index + index_k_fold
        # define test set
        x_test = x[left_index:right_index]
        y_test = y[left_index:right_index]

        # define train set
        if iteration == 1:
            x_train = x[right_index : N - 1]
            y_train = y[right_index : N - 1]

        elif iteration == nb_k_fold:
            x_train = x[0:left_index]
            y_train = y[0:left_index]

        else:
            x_train = np.row_stack((x[0:left_index], x[right_index : N - 1]))
            y_train = concatenate_function((y[0:left_index], y[right_index : N - 1]))

        iteration = iteration + 1
        # modelling
        parameter = Class_algorithm.fit(x_train, y_train)
        y_test_predicted = Class_algorithm.predict(x_test, parameter)
        metric = metrics_function(y_test, y_test_predicted)
        metrics_list.append(metric)
    Class_algorithm.need_to_store_results = True

    return metrics_list

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
from itertools import product
from _base import block_diagonal, block_prob_matrix

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
        #if the introduced y is a pre-transofrmed matrix-> skip this step , otherwise transform 
        #maybe include descending ascending parameter
        #attention, you have to include afterwards which reference class is modelised ( in original y notation)
        if len(y.shape) == 1:
            unique_classes = sorted(np.unique(y))
            ft = True
            #identify number of classes, if logistic then use K-1 classes encoding, if ovr then K classes encoding
            number_classes=len(unique_classes) if self.multiclass=="ovr" else len(unique_classes)-1
            for unique_class in unique_classes[:number_classes]:
                    class_y = np.array([1 if el == unique_class else 0 for el in y])

                    if ft:
                        ft = False

                        result_true_label = class_y
                    else:

                        result_true_label = np.column_stack(
                            (result_true_label, class_y)
                        )

            y = result_true_label.copy()
                
            
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
        
        self.nb_unique_classes=sorted(np.unique(y))
        
        y = self.transform_y_vector_to_matrix(y) if len(np.unique(y))>2 else y 


        if self.multiclass == "binary":
           
            result_param = super().fit_base(x, y, "logistic")

        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        #dont use ovr for the moment, maybe we have to do ovr with original manner? or maybe not? TEST!!!!  
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
          
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
            
            
            #HERE PRIORITY 1
            self.y =y 
            

        return result_param

    @staticmethod
    def __normalise_predictions(proba: np.ndarray) -> np.ndarray:
        """ "
        In case of OVR regression, normalise probabilities
        """
        sum_columns = np.array(([sum(x) for x in zip(*proba.T)]))

        sum_columns = np.column_stack([sum_columns] * proba.shape[1])
       
        return sum_columns

    @staticmethod
    def __from_proba_to_prediction(proba: np.ndarray) -> np.ndarray:
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
     
    
    def __get_proba(self,linear_predictions: np.ndarray)->np.ndarray:
            """  
            Get from linear predictions with softmax function probabilities
            
            """
            numerator = np.exp(linear_predictions)
            
            
            denominator = np.zeros(linear_predictions.shape[0])
            
            nb_cols=linear_predictions.shape[1] if linear_predictions.ndim>1 else linear_predictions.ndim
            
            
            for i in range(nb_cols):
                denominator = denominator + numerator[:, i] if linear_predictions.ndim>1 else denominator + numerator
            if self.multiclass!="ovr":
                denominator=denominator+np.exp(0)#we add reference class
            denominator = np.column_stack([denominator] * linear_predictions.shape[1])  if linear_predictions.ndim>1  else denominator
            
            proba = numerator / denominator                        
            
            return proba
        
    def predict(
        self,
        x: np.ndarray,
        param_if_not_kept: np.ndarray = None,
        threshold: float = 0.5,
        last_class: bool = True
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

        erh.check_arguments_data((x, np.ndarray), (threshold, float), (last_class,bool))
        linear_predictions = super().predict_linear(x, param_if_not_kept)

        proba = self.__get_proba(linear_predictions)
        
        if self.multiclass!="ovr" :
            
            proba_K_class=(1-np.array(([sum(x) for x in zip(*proba.T)]))) if linear_predictions.ndim>1 else 1-proba
           
            proba=np.column_stack(((proba,proba_K_class)))
            
        elif self.multiclass=="ovr":
            proba=proba/ self.__normalise_predictions(proba)
        prediction = self.__from_proba_to_prediction(proba)
        
        # if not last_class:
        len_pred=prediction.shape[1]
        prediction=prediction[:,:-1] if self.multiclass!="ovr" else prediction
        prediction=prediction.reshape(prediction.shape[0],) if len_pred==2 else prediction
        
        

        if self.need_to_store_results:
            self.proba = proba

        return prediction

    
 




    def get_inference(
        self,
        only_IC: bool = False,
        param_if_not_kept: np.ndarray = None,
        y_if_not_kept: np.ndarray = None,
        x_if_not_kept: np.ndarray = None,
        ordered_columns_names : np.ndarray =None
    
    ) -> pd.DataFrame:
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
        if self.multiclass=="ovr" and only_IC==False:
            raise ValueError("for OVR there is no parameter significance, use standard logistic regression")

        if self.need_to_store_results and isinstance(self.proba, str):
            raise ValueError("run .predict() first")
        
        if self.need_to_store_results:
            parameter = self.params
            
            y=self.y
            
               
           
            x=self.x
            proba=self.proba[:,:-1] if self.multiclass!="ovr" else self.proba
            if param_if_not_kept is not None or y_if_not_kept is not None or  x_if_not_kept is not None :
                print("introduced param/y/x ignored, as we store results")
        else:
            #correct it all here!!!!!!!!!!!!
            parameter = param_if_not_kept
            y=y_if_not_kept
            
            x=super().add_intercept_f(x_if_not_kept)
            
            linear_predictions = super().predict_linear(x_if_not_kept, parameter)
            
            
            proba= self.__get_proba(linear_predictions)
            
                
            
            if param_if_not_kept is None or y_if_not_kept is  None or x_if_not_kept is  None:
                raise ValueError(
                    "we dont store results, so you neeed to introduce param"
                )
     
        N = x.shape[0]
        p = x.shape[1]

        if self.y.ndim==1:
           LL = y.T @ x @ parameter- np.sum(
               np.log(1 + np.exp(x @ parameter))
           )
        else:
            #positive cross entropy for OVR and softmax cases
        
            LL=np.sum(np.diag(y@np.log(proba.T)))#!!!!
        
        AIC_ll = -2 * LL + 2 * (p)
        BIC_ll = -2 * LL + np.log(N) * (p)
        
        criteria = {}
        criteria["LL"] = LL
        criteria["AIC_ll"] = AIC_ll
        criteria["BIC_ll"] = BIC_ll
        
        if only_IC:
            return criteria
        
        
        
        
        
        #starting from here -> change to have good inference based on block matrices, finally validate with more classes but seems OK
        nb_dimensions=1 if y.ndim==1 else y.shape[1]
        
        xx=block_diagonal(x,nb_dimensions)
        
        ww=block_prob_matrix(proba,nb_dimensions)
        
        #variance covariance matrix is inverse of hessian 
        # print(np.diag((np.linalg.inv(xx.T@ww@xx))))
        vector_std_errors=np.sqrt(np.diag((np.linalg.inv(xx.T@ww@xx))))
       
        #create data frame with coefficients
        ft=True
        nb_class=1 if parameter.ndim==1 else parameter.shape[1]
        for i in range(nb_class):
            if ft:
                stacked_params=parameter[:,i] if parameter.ndim>1 else parameter
                ft=False
            else:
                stacked_params=np.hstack((stacked_params,parameter[:,i]))
        z_value=stacked_params/vector_std_errors
        p_value=np.array([(2 * t.sf(np.abs(el), df=N - p)) for el in z_value])
        p_value_star=np.array(["***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else "" for p_value in p_value])
        odds_ratio=np.exp(stacked_params)
        result=np.column_stack((np.round(stacked_params,decimals=4),np.round(vector_std_errors,decimals=2),np.round(z_value,decimals=2),np.round(p_value,decimals=3),p_value_star,np.round(odds_ratio,decimals=2)) )
        result_column_names=["coef","std","z value","p value","p value star","odds ratio"]
        
        if ordered_columns_names!=None: 

            
            ordered_columns_names=np.insert(ordered_columns_names, 0, "intercept") if self.add_intercept else ordered_columns_names
        #attention! in ALL my work i work with labels 0,1,2,3 etc!!!another case is not accepted!! can be fixed, but i don't have time
            if y.ndim==2:
                #take all except for the last class which is the class of reference
                classes_modeled=self.nb_unique_classes[0:len(self.nb_unique_classes)-1]
            else:
                classes_modeled=[1]
                
            ft=True
            for el in classes_modeled:
                if ft:
                    col_names=np.array([f"{col}_{el}" for col in ordered_columns_names])
                   
                    ft=False
                else:
                    col_names=np.hstack((col_names,np.array([f"{col}_{el}" for col in ordered_columns_names])))
            ordered_columns_names=col_names   
            
        df = pd.DataFrame(result,index=ordered_columns_names,columns=result_column_names)
        
        
        
           
        
        if self.need_to_store_results:
            self.criteria = criteria
        # calculate variance covariance matrix and p values of parameters

        # t_value = self.params / std_params
        # p_value = [(2 * t.sf(np.abs(el), df=N - p)) for el in t_value]
        # result = pd.DataFrame(
        #     np.column_stack((self.params, std_params, t_value, np.round(p_value, 4))),
        #     columns=["params", "std", "t value", "p value"],
        # )
        return df
    
    
    def autoselection(
        self,
        method: Literal["backward", "forward", "stepwise"],
        criterion: Literal[
            "BIC_ll",
            "AIC_ll",
          
            "LL",
        ] = "BIC_ll",
        print_message: bool=True
    )->np.ndarray:
        
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
        
        
        result=super().autoselection(method,criterion,print_message)
        return result
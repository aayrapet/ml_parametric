"""
Variable Auto Selection using Information Criteria
"""

# Author: Artur Ayrapetyan

import numpy as np

import math as mt
from typing import Type
from typing import Literal
import _err_handl as erh


class AutoSelect:
    def __init__(
        self,
        Class_algorithm: Type,
        method: Literal["backward", "forward", "stepwise"],
        inf_criterion: Literal[
            "BIC_ll",
            "AIC_ll",
            "AIC_err",
            "BIC_err",
            "LL",
        ] = "BIC_ll"
    ):  
        """
        Initializes an AutoSelect object.

        Parameters:
        ----
            Class_algorithm (Type): The type of the algorithm class to be used.
            method (Literal["backward", "forward", "stepwise"]): The selection method to be used. 
                It can be one of the following: "backward", "forward", or "stepwise".
            inf_criterion (Literal["BIC_ll", "AIC_ll", "AIC_err", "BIC_err", "LL"], optional):
                The information criterion to be used for model selection. 
                Default is "BIC_ll".

        Backward Regression:
        ---

        In backward regression, you start with all the predictors (independent variables) included in the model, 
        and then iteratively remove the least significant predictor one at a time until a stopping criterion is met. 
        This stopping criterion is often based on statistical tests like p-values or information criteria such as AIC 
        (Akaike Information Criterion) or BIC (Bayesian Information Criterion). 
        Backward regression is beneficial when dealing with a large number of predictors and you want to simplify the model by removing irrelevant variables.
        
        Forward Regression:
        -----

        Unlike backward regression, forward regression starts with an empty model and iteratively adds the most significant
        predictor one at a time until a stopping criterion is met. Similar to backward regression, the stopping criterion 
        is often based on statistical tests or information criteria. Forward regression is useful when you have a large pool 
        of potential predictors and you want to identify the most relevant variables for your model.
        
        Stepwise Regression:
        -------
        Stepwise regression  combines aspects of both backward and forward regression.
        It starts with an empty model and at each step, it either adds or removes a predictor based on predefined criteria.
       
        """
        #check that data types are good 
        erh.check_arguments_data((Class_algorithm,"__class__"),(method,str),(inf_criterion,str))
        self.Class_algorithm=Class_algorithm
        self.method = method
        self.inf_criterion = inf_criterion
        
        

    @staticmethod
    def is_decrease(vector: list) -> bool:
        """
        This function is used for model selection using Information Criteria (IC)
        For this, we will observe only two last values of the list. The selection algorithm will have 3 rules:

        1.iF N-1 value > N value then continue the algorithm so the function will return True
        2.if there is no values yet in the list, so we work with initialised vector so the function will return True to start accuulating IC
        3.if the lenght of the function is 1 return True to get more IC for model comparison

        once N-1 value<N value the function returns False, the algorithm stops -> we get our final model that minimses IC


        """
        val = False
        if not vector:
            val = True
        elif len(vector) == 1:
            val = True
        elif len(vector) >= 2:
            if (vector[len(vector) - 2] - vector[len(vector) - 1]) > 0:
                val = True
        return val

    @staticmethod
    def delete_i_from_index(exclusion: int, vector: np.ndarray) -> np.ndarray:
        """

        Simple function just to exclude observation from numpy array
        exclusion: digit to exclude
        vector: exclude from this numpy array this digit


        """
        new_index = []
        for i in range(len(vector)):
            if vector[i] != exclusion:
                new_index.append(vector[i])
        return new_index

    def forward_selection(self,
        x: np.ndarray,
        y: np.ndarray,
        
    ) -> np.ndarray:
        """
        Forward selection Algorithm

        Starts with empty model and iteratively adds variables to the model whose IC is the least

        Parameters
        ------
        
        x (array alike) : matrix of x variables
        y (array alike) : vector of y
        criterion (str) : Information criterion (IC) that is a stopping criterion that stops

                For linear regression accepted are:
                                     -BIC_ll |AIC_ll |AIC_err |BIC_err |LL
                For logistic regression accepted are:
                                     -BIC_ll |AIC_ll |LL because we cant calculate errors




        """
        # check that introduced variables are of good type
        erh.check_arguments_data(
           
            (x, np.ndarray),
            (y, np.ndarray),
         
        )
        self.Class_algorithm.need_to_store_results=False
       

        index = np.array([i for i in range(x.shape[1])])

        min_aic_global = []
        first_iteration = True
        final_index = np.zeros(0, dtype=int)
        index_found = 0

        while self.is_decrease(min_aic_global):
            if first_iteration:
                first_iteration = False
            else:
                # delete from index respective variable
                index = self.delete_i_from_index(index_found, index)
                # add to final index respective variable
                final_index = np.hstack((final_index, index_found))

            min_aic = []
            for i in range(len(index)):
                new_index = np.hstack((final_index, index[i]))
                #run model on selected set of variables
                parameter = self.Class_algorithm.fit(x[:, new_index], y)
                criteria = self.Class_algorithm.get_inference(
                    #only calculate IC
                    only_IC=True,
                    #as we dont store parameters/x/y, introduce them in arguments!!
                    param_if_not_kept=parameter,
                    y_if_not_kept=y,
                    x_if_not_kept=x[:, new_index],
                )
                this_criterion = criteria[self.inf_criterion]
                min_aic.append(this_criterion)
            # find position of minimal IC over these models
            index_min = np.argmin(min_aic)
            index_found = index[index_min]
            # add minimal IC to the IC list, so that we can compare two last IC
            min_aic_global.append(min(min_aic))
        # set this parameter back to true so the user will be able to store results in attributes
        self.Class_algorithm.need_to_store_results = True

        return final_index

    def backward_selection(
        self,
        x: np.ndarray,
        y: np.ndarray,
       
    ) -> np.ndarray:
        """
        Backward selection Algorithm

        Starts with full model and iteratively removes variables from the model whose IC is the least

        Parameters
        ------
        
        x (array alike) : matrix of x variables
        y (array alike) : vector of y
        criterion (str) : Information criterion (IC) that is a stopping criterion that stops

                For linear regression accepted are:
                                     -BIC_ll |AIC_ll |AIC_err |BIC_err |LL
                For logistic regression accepted are:
                                     -BIC_ll |AIC_ll |LL because we cant calculate errors




        """
        # check that introduced variables are of good type
        erh.check_arguments_data(
           
            (x, np.ndarray),
            (y, np.ndarray),
          
        )
        # make sure that during code excecution we dont store the process of this function
        self.Class_algorithm.need_to_store_results = False
        index = np.array([i for i in range(x.shape[1])])

        first_iteration = True

        index_found = 0

        while True:
            if first_iteration:
                first_iteration = 0
            else:
                index = self.delete_i_from_index(index_found, index)

            start = float("inf")
            first_time = True
            # /* run first the whole model-> then extract variables -> start with O (whole model)   */
            for i in range(-1, len(index)):
                if first_time:
                    new_index = index.copy()
                    first_time = False
                else:
                    new_index = self.delete_i_from_index(index[i], index)
                #run model on selected vaiables of x 
                parameter = self.Class_algorithm.fit(x[:, new_index], y)
                criteria = self.Class_algorithm.get_inference(
                    #only calculate IC
                    only_IC=True,
                    #as we dont store parameters/x/y, introduce them in arguments!!
                    param_if_not_kept=parameter,
                    y_if_not_kept=y,
                    x_if_not_kept=x[:, new_index],
                )
                this_criterion = criteria[self.inf_criterion]

                if start > this_criterion:
                    start = this_criterion
                    if i == -1:
                        index_found = "Full model is best"
                    else:
                        index_found = index[i]

            # the exit from the loop is guaranteed when index_found is "Full model is best"
            if isinstance(index_found, str):
                self.Class_algorithm.need_to_store_results = True
                return index

    def stepwise_selection(
        self,
       
        x: np.ndarray,
        y: np.ndarray,
        
    ) -> np.ndarray:
        """
        Stepwise selection Algorithm

        Starts with  model filled with 50% of variables selected randomly (inside vars),
        other 50% are kept outside the model (outide vars).

        Tries to remove variables from inside vars doing backward regression
        on them-> calculates the min IC and respective inside var

        At the same time tries to add variables to inside vars doing forward regression
        and calculates minimal IC and respective inside var.

        The alg will drop the variable if min IC backward < min IC forward and vice versa.
        Very important: when the variable is removed, it does not leave the model forever,
        it will go outside vars so that we leave the possibility to this var to get back later.

        Parameters
        ------
        Class_algorithm (class) : parametric model on which variable selection will be performed
        x (array alike) : matrix of x variables
        y (array alike) : vector of y
        criterion (str) : Information criterion (IC) that is a stopping criterion that stops

                For linear regression accepted are:
                                     -BIC_ll |AIC_ll |AIC_err |BIC_err |LL
                For logistic regression accepted are:
                                     -BIC_ll |AIC_ll |LL because we cant calculate errors




        """
        # check that introduced variables are of good type
        erh.check_arguments_data(
           
            (x, np.ndarray),
            (y, np.ndarray),
          
        )
        # make sure that during code excecution we dont store the process of this function

        self.Class_algorithm.need_to_store_results = False
        v = np.random.permutation(x.shape[1])
        split_index = mt.floor(x.shape[1] / 2)
        index = v[0:split_index]
        remaining_index = v[split_index:]

        min_aic = []
        first_iteration = True

        while self.is_decrease(min_aic):
            if first_iteration:

                first_iteration = False

            else:
                if min_remove_criterion < min_add_criterion:
                    index = self.delete_i_from_index(index_found_remove, index)
                    remaining_index = np.hstack((remaining_index, index_found_remove))
                else:
                    remaining_index = self.delete_i_from_index(
                        index_found_add, remaining_index
                    )
                    index = np.hstack((index, index_found_add))

            start = float("inf")
            first_time = True
            # /*BACKWARD regression*/
            if len(index) != 0:
                for i in range(-1, len(index)):
                    if first_time:
                        new_index = index.copy()
                        first_time = False

                    else:
                        new_index = self.delete_i_from_index(index[i], index)

                    parameter = self.Class_algorithm.fit(x[:, new_index], y)
                    criteria = self.Class_algorithm.get_inference(
                        #only calculate IC
                        only_IC=True,
                        #as we dont store parameters/x/y, introduce them in arguments!!
                        param_if_not_kept=parameter,
                        y_if_not_kept=y,
                        x_if_not_kept=x[:, new_index],
                    )
                    this_criterion = criteria[self.inf_criterion]

                    if start > this_criterion:
                        start = this_criterion
                        # if i == -1:
                        #     #get back here!!
                        #     index_found = "Full model is best"
                        # else:
                        #     index_found_remove = index[i]
                        index_found_remove = index[i]

                min_remove_criterion = start.copy()
            else:
                # if vector is empty so just do forward selection because min_add_criterion will be  < min_remove_criterion
                min_remove_criterion = min_add_criterion + 1

            # /*FORWARD regression*/

            start = float("inf")
            # /* do all candidates for inclusion (if they still remain) */
            if len(remaining_index) != 0:
                for i in range(len(remaining_index)):

                    new_index = np.hstack((index, remaining_index[i]))

                    parameter = self.Class_algorithm.fit(x[:, new_index], y)
                    criteria = self.Class_algorithm.get_inference(
                        #only calculate IC
                        only_IC=True,
                        #as we dont store parameters/x/y, introduce them in arguments!!
                        param_if_not_kept=parameter,
                        y_if_not_kept=y,
                        x_if_not_kept=x[:, new_index],
                    )
                    this_criterion = criteria[self.inf_criterion]

                    if start > this_criterion:
                        start = this_criterion.copy()
                        index_found_add = remaining_index[i]

                min_add_criterion = start.copy()

            else:
                # if vector is empty so just do backward selection because min_remove_criterion will be  < min_add_criterion
                min_add_criterion = min_remove_criterion + 1

            if min_remove_criterion < min_add_criterion:
                min_aic.append(min_remove_criterion)
            else:
                min_aic.append(min_add_criterion)

        self.Class_algorithm.need_to_store_results = True
        return index

import numpy as np
from _linear import LinearRegression
import math as mt 

def is_decrease(vector):
    val = False
    if not vector:
        val = True
    elif len(vector) == 1:
        val = True
    elif len(vector) >= 2:
        if (vector[len(vector) - 2] - vector[len(vector) - 1]) > 0:
            val = True
    return val


def delete_i_from_index(exclusion, vector):
    new_index = []
    for i in range(len(vector)):
        if vector[i] != exclusion:
            new_index.append(vector[i])
    return new_index


def forward_selection(Class_algorithm, x, y, criterion):

    Class_algorithm.need_to_store_results = False
    index = np.array([i for i in range(x.shape[1])])

    min_aic = []
    first_iteration = True
    final_index = np.zeros(0, dtype=int)
    index_found = 0

    while is_decrease(min_aic):
        if first_iteration:
            first_iteration = False
        else:
            index = delete_i_from_index(index_found, index)
            final_index = np.hstack((final_index, index_found))
        start = float("inf")

        for i in range(len(index)):
            new_index = np.hstack((final_index, index[i]))

            parameter = Class_algorithm.fit(x[:, new_index], y)
            criteria = Class_algorithm.get_inference(
                only_IC=True,
                param_if_not_kept=parameter,
                y_if_not_kept=y,
                x_if_not_kept=x[:, new_index],
            )
            this_criterion = criteria[criterion]

            if start > this_criterion:
                start = this_criterion.copy()
                index_found = index[i]

        min_aic.append(start)
    Class_algorithm.need_to_store_results = True
    return final_index


def backward_selection(Class_algorithm, x, y, criterion):
    Class_algorithm.need_to_store_results = False
    index = np.array([i for i in range(x.shape[1])])

    min_aic = []
    first_iteration = True

    index_found = 0

    # continue if aic continue decreasing or if the minimal aic is not the whole model
    while is_decrease(min_aic):
        if first_iteration:
            first_iteration = 0

        else:
            index = delete_i_from_index(index_found, index)

        start = float("inf")
        first_time = True
        # /* run first the whole model-> then extract variables -> start with O (whole model)   */
        for i in range(-1, len(index)):
            if first_time:
                new_index = index.copy()
                first_time = False

            else:
                new_index = delete_i_from_index(index[i], index)

            parameter = Class_algorithm.fit(x[:, new_index], y)
            criteria = Class_algorithm.get_inference(
                only_IC=True,
                param_if_not_kept=parameter,
                y_if_not_kept=y,
                x_if_not_kept=x[:, new_index],
            )
            this_criterion = criteria[criterion]

            if start > this_criterion:
                start = this_criterion
                if i == -1:
                    index_found = "Full model is best"
                else:
                    index_found = index[i]

        min_aic.append(start)
        if isinstance(index_found, str):
            Class_algorithm.need_to_store_results = True
            return index
    Class_algorithm.need_to_store_results = True
    return index


def stepwise_selection(Class_algorithm, x, y, criterion):
    Class_algorithm.need_to_store_results = False
    v = np.random.permutation(x.shape[1])
    split_index = mt.floor(x.shape[1] / 2)
    index = v[0:split_index]
    remaining_index = v[split_index:]

    min_aic = []
    first_iteration = True

    while is_decrease(min_aic):
        if first_iteration:

            first_iteration = False

        else:
            if min_remove_criterion < min_add_criterion:
                index = delete_i_from_index(index_found_remove, index)
                remaining_index = np.hstack((remaining_index, index_found_remove))
            else:
                remaining_index = delete_i_from_index(index_found_add, remaining_index)
                index = np.hstack((index, index_found_add))

        start = float("inf")
        first_time = True
        # /*BACKWARD regression*/
        for i in range(-1, len(index)):
            if first_time:
                new_index = index.copy()
                first_time = False

            else:
                new_index = delete_i_from_index(index[i], index)

            parameter = Class_algorithm.fit(x[:, new_index], y)
            criteria = Class_algorithm.get_inference(
                only_IC=True,
                param_if_not_kept=parameter,
                y_if_not_kept=y,
                x_if_not_kept=x[:, new_index],
            )
            this_criterion = criteria[criterion]

            if start > this_criterion:
                start = this_criterion
                if i == -1:
                    index_found = "Full model is best"
                else:
                    index_found_remove = index[i]

        min_remove_criterion = start.copy()
        # /*FORWARD regression*/

        start = float("inf")
        # /* do all candidates for inclusion (if they still remain) */
        if len(remaining_index) != 0:
            for i in range(len(remaining_index)):

                new_index = np.hstack((index, remaining_index[i]))

                parameter = Class_algorithm.fit(x[:, new_index], y)
                criteria = Class_algorithm.get_inference(
                    only_IC=True,
                    param_if_not_kept=parameter,
                    y_if_not_kept=y,
                    x_if_not_kept=x[:, new_index],
                )
                this_criterion = criteria[criterion]

                if start > this_criterion:
                    start = this_criterion.copy()
                    index_found_add = remaining_index[i]

            min_add_criterion = start.copy()

        else:  # ?
            min_add_criterion = min_remove_criterion + 1

        if min_remove_criterion < min_add_criterion:
            min_aic.append(min_remove_criterion)
        else:
            min_aic.append(min_add_criterion)

    Class_algorithm.need_to_store_results = True
    return index

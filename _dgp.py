"""
Data Generation Process

"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import rankdata, norm
from scipy.linalg import cholesky


def ImanConoverTransform(X, C):
    # used for outliers generation
    N = X.shape[0]
    S = np.ones_like(X)

    for i in range(X.shape[1]):
        ranks = rankdata(X[:, i], method="average")
        S[:, i] = norm.ppf(ranks / (N + 1))

    CS = np.corrcoef(S, rowvar=False)
    Q = cholesky(CS, lower=True)
    P = cholesky(C, lower=True)
    T = np.linalg.solve(Q, P)
    Y = np.dot(S, T)

    W = X.copy()
    for i in range(Y.shape[1]):
        rank = rankdata(Y[:, i], method="ordinal")
        tmp = W[:, i].copy()
        tmp.sort()
        W[:, i] = tmp[rank - 1]

    return W


def gen(type=1):
    if type == 1:
        # Define covariance matrix
        var_covar = np.eye(50)

        # Set mean vector
        mean = np.zeros((1, var_covar.shape[1]))

        # Generate the coefficient vector
        beta = np.array(
            [2.5, 1.3, 0.5, -0.5, -3.4]
        )  #  np.random.uniform(-10, 10, size=5)/3 #

        # Generate random normal variables
        x = np.random.multivariate_normal(mean.flatten(), var_covar, 500)

        # Generate epsilon values
        eps = np.random.normal(loc=0, scale=0.1, size=x.shape[0])

        # Create Y variable
        y = (x[:, :5] @ beta.T) + eps
    elif type == 2:
        var_covar1 = np.full((10, 10), 0.9)  # X1 to X10 strongly correlated: 0.9
        var_covar2 = np.full((10, 10), 0.8)  # X11 to X20
        var_covar3 = np.full((10, 10), 0.7)  # X21 to X30
        var_covar4 = np.full((10, 10), 0.6)  # X31 to X40
        var_covar5 = np.full((10, 10), 0.5)  # X41 to X50

        # Ensure there are only 1s on the diagonal
        for i in range(10):
            var_covar1[i, i] = 1
            var_covar2[i, i] = 1
            var_covar3[i, i] = 1
            var_covar4[i, i] = 1
            var_covar5[i, i] = 1  # Diagonal to 1 for variance

        mean = np.zeros(50)  # mean = 0 vector
        beta = np.array([2.5, 1.3, 0.5, -0.5, -3.4])  # generate the coefficient vector

        X1 = np.random.multivariate_normal(
            mean[0:10], var_covar1, 500
        )  # Generate 500 observations for X1 to X10 with correlation of 0.9
        X2 = np.random.multivariate_normal(
            mean[10:20], var_covar2, 500
        )  # Generate 500 observations for X11 to X20 with correlation of 0.8
        X3 = np.random.multivariate_normal(
            mean[20:30], var_covar3, 500
        )  # Generate 500 observations for X21 to X30 with correlation of 0.7
        X4 = np.random.multivariate_normal(
            mean[30:40], var_covar4, 500
        )  # Generate 500 observations for X31 to X40 with correlation of 0.6
        X5 = np.random.multivariate_normal(
            mean[40:50], var_covar5, 500
        )  # Generate 500 observations for X41 to X50 with correlation of 0.5

        x = np.concatenate((X1, X2, X3, X4, X5), axis=1)
        eps = np.random.normal(0, 0.1, size=x.shape[0])
        y = (x[:, :5] @ beta) + eps

    elif type == 3:

        var_covar = np.array(
            [
                [1, 0.6, 0.8, 0.49, 0.64],
                [0.6, 1, 0.73, 0.46, 0.51],
                [0.8, 0.73, 1, 0.66, 0.62],
                [0.49, 0.46, 0.66, 1, 0.48],
                [0.64, 0.51, 0.62, 0.48, 1],
            ]
        )

        var_covar_id = np.eye(45)  # identity matrix for the remaining 45 variables
        mean = np.zeros(50)  # mean vector of zeros for all variables
        beta = np.array([2.5, 1.3, 0.5, -0.5, -3.4])

        X_corr = multivariate_normal.rvs(mean=mean[:5], cov=var_covar, size=500)

        X_indep = multivariate_normal.rvs(mean=mean[5:], cov=var_covar_id, size=500)

        x = np.concatenate((X_corr, X_indep), axis=1)

        eps = np.random.normal(loc=0, scale=0.1, size=x.shape[0])

        y = np.dot(x[:, :5], beta) + eps
    elif type == 4:
        N = 500
        mu1 = np.zeros(50)
        cov1 = np.eye(50)
        X1 = np.random.multivariate_normal(mu1, cov1, N)
        mu2 = np.ones(50) * 5
        cov2 = np.eye(50)
        X2 = np.random.multivariate_normal(mu2, cov2, N)

        p = 0.9
        U = np.random.uniform(size=N)
        X = np.zeros((N, 50))

        for i in range(50):
            for j in range(N):
                if np.random.uniform() < p:
                    X[j, i] = X1[j, i]
                else:
                    X[j, i] = X2[j, i]

        R = X[:, :5]
        M = X[:, 5:]

        C = np.array(
            [
                [1, 0.6, 0.8, 0.49, 0.64],
                [0.6, 1, 0.73, 0.46, 0.51],
                [0.8, 0.73, 1, 0.66, 0.62],
                [0.49, 0.46, 0.66, 1, 0.48],
                [0.64, 0.51, 0.62, 0.48, 1],
            ]
        )

        W = ImanConoverTransform(R, C)
        x = np.concatenate((W, M), axis=1)

        beta = np.array([2.5, 1.3, 0.5, -0.5, -3.4])
        eps = np.random.normal(size=N) * 0.1
        y = np.dot(x[:, :5], beta) + eps
    return x, y

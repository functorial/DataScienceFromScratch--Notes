#
# Multiple Regression
#

# Sometimes just called:
#   Linear Regression
#
# Like Simple Linear Regression, except
#   each input x_i is a vector
# 
# The model:
#   (fixed inputs):     k-vector:   x_i = [1, x_i1, x_i2, ..., x_ik]        # Think: x_i is a person, each field x_ij is a measurement
#   (fixed outputs):    float:      y_i
#   (linear coefs):     k-vector:   beta = [b_0, b_1, ..., b_k]
#   (error):            float:      e_i 
#
#   Assume:
#       y_i = b_0 + b_1*x_i1 + ... + b_k*x_ik + e_i
#           = beta.T @ x_i + e_i
#   Assume:
#       The columns [x_i1, x_i2, ..., x_ik] of x are linearly independent
#           Otherwise we can't solve for beta
#   Assume:
#       The columns of x are uncorrelated with the errors e
#           Otherwise, beta will be biased towards a systematically wrong value
#
#   Treat e_i as function of beta:
#       e_i = y_i - beta.T @ x_i
#
#   Pick beta so that the OLS (Ordinary Least Squares) is minimized:
#       model_sq_error = sum( (y_i - beta.T @ x_i)**2 for x_i, y_i in zip(xs, ys) )
#
#   Then predict:
#       predict(x) == beta.T @ x


# Can use gradient descent to solve, but don't!!!
#   There exists an analytical solution!!!!:
#       beta = np.reciprocal( np.linalg.norm(x) ) * (x.T @ y)

import numpy as np

# get beta
def lin_reg_coeffs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """computes the multiple linear regression coefficients of the data"""
    # append 1 to the beginning of x_i for all i
    x1 = np.insert(xs, 0, 1, axis=1)
    P = np.linalg.norm(x1) ** -1 * x1.T
    beta = P @ ys
    return beta

# use to build model function
def lin_reg_predict(x_i: np.ndarray, beta: np.ndarray) -> float:
    """x_i is a vector of the independent data, beta are the regression coeffs. 
    returns y_i based on beta""" 
    x1 = np.insert(x_i, 0, 1)
    return np.dot(x1, beta)

# residual of an observed value against the model
def lin_reg_error(x_i: np.ndarray, y_i: float, beta: np.ndarray) -> float:
    """residual of an observed value measured against the model"""
    return y_i - lin_reg_predict(x_i, beta)

# unexplained varation
def residual_sum_of_squares(xs: np.ndarray, ys: np.ndarray, beta: np.ndarray) -> float:
    return sum( lin_reg_error(x_i, y_i, beta)**2 for x_i, y_i in zip(xs, ys) )

# total variation
def total_sum_of_squares(ys: np.ndarray) -> float:
    ys_bar = ys.mean()
    return np.sum( np.square(ys - ys_bar) )

# R-squared of the model
def lin_reg_r_squared(xs: np.ndarray, ys: np.ndarray, beta: np.ndarray) -> float:
    return 1 - residual_sum_of_squares(xs, ys, beta) / total_sum_of_squares(ys)

# Problem:
#   Adding new variables to a regression will necessarily increase the R-squared
# Solution:
#   None, look at more statistics.
#   
#   In particular, look at the `standard errors` of the beta_i
#       Measure how certain we are about each beta_i
# 

# Standard Errors of Regression Coefficients

# Assume:
#   The errors are normally distributed RVs with mean=0, and share a variance
#   
# We can use bootstrapping to measure our confidence in our regression coefficients
from typing import Tuple, List
from Bootstrap import bootstrap_statistic

def estimate_sample_beta(pairs: List[Tuple[np.ndarray, float]]):
    x_sample = np.ndarray([x for x, _ in pairs])
    y_sample = np.ndarray([y for _, y in pairs])
    beta = lin_reg_coeffs(x_sample, y_sample)
    print("bootstrap sample", beta)
    return beta

# Suppose we have supervised data xs, ys
#
# Generate bootstrap samples and compute beta on them
#   bootstrap_betas = bootstrap_statistic( list(zip(xs, ys)), estimate_sample_beta, 100)
#
# Estimate standard deviation of each coefficient
#   bootstrap_standard_errors = [np.std(np.array( [beta[i] for beta in bootstrap_betas] )) 
#                                                               for i in range(len(beta))]

# or....
# Analytic solution?
#   Copied off of wikipedia, may have errors
def std_errors_of_coeffs(xs: np.ndarray, ys: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = ys.size
    p = beta.size
    reduced_chi_squared = residual_sum_of_squares(xs, ys, beta) / (n - p)
    est_covariance = reduced_chi_squared * np.linalg.inv(xs.T @ xs)
    size = est_covariance.size[0]
    return np.array([ np.sqrt( est_covariance[i][i] ) for i in range(size) ])


#
# Regularization
#
# Would like to apply linear regression to high-dimensional data
#   Problems:
#       (1) More dimensions = more chance to overfit the data
#       (2) Harder to explain phenomena (if that is your goal)
#   Solution:
#       Regularization
#           (*) Add a penalty term to OLS error which grows as the 'size of beta' grows,
#               Pick beta which minimizes the new objective function
#       
#       Ridge Regression:
#           A type of regularizaton:
#               The penalty is proportional to norm(beta)
#                   Sometimes don't penalize constant term...

# alpha = 'hyper-parameter':
#   controls how harsh the penalty is
def ridge_penalty(beta: np.ndarray, alpha: float) -> float:
    return alpha * np.linalg.norm(beta[1:]) # don't want to penalize constant term

# want to pick beta that minimizes this
def sq_error_ridge(xs: np.ndarray, ys: np.ndarray, beta: np.ndarray, alpha: float) -> float:
    """estimate OLS plus ridge penalty on beta"""
    return residual_sum_of_squares(xs, ys, beta) + ridge_penalty(beta, alpha)

# Analytic solution?
#   Can solve with gradient descent....

def _resid_sum_sq_gradient(xs: np.ndarray, ys: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return  2 * xs.T @ (ys - xs @ beta)

def _ridge_penalty_gradient(beta: np.ndarray, alpha: float) -> np.ndarray:
    return np.insert(2* alpha * beta, 0, 0)     # change in constant term does not affect penalty

def _sq_error_ridge_gradient( xs: np.ndarray, ys: np.ndarray, beta: np.ndarray, alpha: float) -> float:
    return _resid_sum_sq_gradient(xs, ys, beta) + _ridge_penalty_gradient(beta, alpha)

import tqdm
import random

# Warning: 
#   make sure the data is rescaled to be unitless,
#   otherwise, there could be biases
# 
def OlS_ridge_fit(xs: np.ndarray, ys: np.ndarray, alpha: float,
                  learning_rate: float, epochs: int, batch_size: int) -> np.ndarray:
    """Returns beta which is a local minimum of sq_error_ridge"""

    guess = np.array( [random.random() for _ in range(len(xs[0]))] )
    starts = range(0, len(xs), batch_size)

    for _ in tqdm.trange(epochs, desc="least squares fit:"):
        # partition inputs into batches
        for start in random.shuffle(starts):
            batch_xs = xs[start : start + batch_size]
            batch_ys = ys[start : start + batch_size]
            guess = guess - learning_rate * _sq_error_ridge_gradient(batch_xs, batch_ys, guess, alpha)
        
    return guess
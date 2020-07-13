#
# Logistic Regression
#

# Linear regression can fail if the output variable is a 'one or the other' type
#   'one or the other' is enoded by 0 and 1, but linear models
#       (1) have unbounded outputs, hard to interpret
#       (2) will be biased since error is correlated with variables
#               large variable = large beta_i = large error from 0 or 1
#   Note:
#       Can generalize this method to output variables which are 'one of k things' type,
#           (i.e. the output variable is categorical)
#           but it is more complicated.
#
# The Logistic Function:
#
#   expit(x) = 1 / (1 + e^(-x))
#   
#   Properties:
#
#       (1) lim(expit(x), x ->  infty) = 1      Solves issue of unboundedness
#           lim(expit(x), x -> -infty) = 0      in a coherent way
#        
#       (2) expit'(x) = expit(x) * (1 - expit(x))
#
#   We would like to have a model where we 
#   push a Linear Regression model
#   through the logistic function:
#
#       log_reg_predict(x) = expit( lin_reg_predict(x) )
#                          = expit( dot( beta, x ) )          for some choice of beta
#
#       (*) Interpret the outcome as a probability:
#               Given an input x, 
#               we predict its output y to be
#                   1   with probability      log_reg_predict(x)
#                   0   with probability  1 - log_reg_predict(x)
#           (*) Note:
#               We are assuming:
#                   A linear relationship between
#                   the log-odds of the event that y = 1 and the predictor variables:
#                   
#                   Let p = P(y=1)
#
#                   log( p / (1-p) ) = dot( beta, x )
#               =>  ...
#               =>  p = 1 / (1 + exp( -dot( beta, x) ) )
#                     = expit( dot( beta, x ) )       
# 
#       (*) Therefore, the Probability Density Function of y is
#               
#               PDF(y) = P( y | beta, x )
#                      = expit( dot(beta, x) )^y * (1 - expit( dot(beta, x) ))^(1 - y)
#
#           Consider as a function of beta, i.e. consider the 'likelihood function':
#               L(beta|x, y) = expit( dot(beta, x) )^y * (1 - expit( dot(beta, x) ))^(1 - y)
#
#   Assume that for every observed input/output (xi, y_i), we have:
#       
#       y_i = expit( dot(x_i, beta) ) + e_i
#
#   With linear regression, we would pick beta that minimizes the OLS error function
#   
#   It turns out it's easier to minimize the 'log likelihood function'
#
#       log( L( beta | xs, ys ) ) =   y * log( expit( dot(beta, x) ) ) + (1 - y) * log( 1 - expit( dot(beta, x) ) )
#       
#       Note:
#           Since log is increasing, maximizing log-likelihood also maximizes likelihood
#   
#   The log-likelihood of multiple data points is the sum of the individual log-likelihoods
#       since the likelihood from multiple data points is the product of the individual likelihoods
#
#   We will pick beta which /maximizes/ the log likelihood. 
#
#   NOTE: log-likelihood is a 'strictly concave function', and thus has exactly one local=global maximum!
#
#   Methods to minimize:
#       (1) Newton's method
#       (2) Gradient descent
#
#   We will go with (2) for now...
#
#   NOTE: Remember to check goodness of fit afterwards


import random
import tqdm
import numpy as np
from scipy.special import expit as expit
from typing import Tuple

def split_zipped_data(xs: np.ndarray, ys: np.ndarray, p: float) -> Tuple(np.ndarray):
    cutoff_index = int(p * xs.shape[0])
    zipped = np.column_stack((xs, ys))
    return zipped[:cutoff_index], zipped[cutoff_index:]


def _log_likelihood_point(x: np.ndarray, y: float, beta: np.ndarray) -> float:
    """Computes the log-likelihood for one datapoint"""
    return y * np.log(expit(np.dot(beta,x))) + (1-y) * np.log(1-expit(np.dot(beta,x)))


def log_likelihood( xs: np.ndarray, ys: np.ndarray, beta: np.ndarray) -> float:
    z = np.column_stack((xs, ys))
    result = 0
    for pair in z:
        x_i = pair[:-1]
        y_i = pair[-1]
        result += _log_likelihood_point(x_i, y_i, beta)
    return result


class Logistic_Regression:
    """A container for logistic regression fitting"""
    def __init__(self, 
                 xs: np.ndarray, 
                 ys: np.ndarray,
                 beta: np.ndarray = np.array([random.random() for _ in xs[0]])):
        self.xs = xs
        self.ys = ys
        self.zip = np.column_stack((self.xs, self.ys)) 
        self.beta = beta

    # define a decision boundary
    def predict_point(self, x: np.ndarray, decision: float=0.5) -> int:
        p = np.dot(self.beta, x)
        if p >= 0:
            return 1
        else:
            return 0

    def _log_likelihood_gradient(self, xs_sample, ys_sample, beta) -> np.ndarray:
        n = np.shape(xs_sample)[0]
        result = np.zeros(n)
        sample_zip = np.column_stack((xs_sample, ys_sample))
        for row in sample_zip:
            y_i = row[:,-1]
            x_i = row[:,:-1]
            result += y_i - expit(np.dot(x_i, beta)) * x_i
        return result


    def _log_likelihood_hessian(self, xs_sample, beta) -> np.ndarray:
        n = np.shape(xs_sample)[0]
        result = np.zeros((n,n))
        for x_i in xs_sample:
            result -= expit(np.dot(beta, x_i)) * (1-expit(np.dot(beta, x_i))) * (x_i @ x_i.T)
        return result        

            
    def train_beta_log_likelihood_gd(self, train_xs: np.ndarray,
                                           train_ys: np.ndarray,
                                           minibatches: int=1,
                                           epochs: int=100, 
                                           step_size: float=0.1) -> None:
        """Crude gradient descent applied to log-likelihood. Updates self.beta."""

        minibatch_size = np.ceil( np.shape(train_xs)[0] / minibatches)
        print(f"Usual minibatch size: {minibatch_size}")
        train_zipped = np.column_stack((train_xs, train_ys))
        minibatches = [train_zipped[n: n + minibatch_size] for n in range(0, train_zipped.shape[0], step=minibatch_size)]

        for _ in tqdm.trange(epochs, desc="log-likelihood fit"):
            for batch in minibatches:
                xs_sample = batch[:, :-1]
                ys_sample = batch[:,-1]
                grad = self._log_likelihood_gradient(xs_sample, ys_sample, self.beta)
                self.beta += step_size * grad
    

    def train_beta_log_likelihood_nm(self, train_xs: np.ndarray,
                                           train_ys: np.ndarray,
                                           minibatches: int=1,
                                           epochs: int=100, 
                                           step_size: float=0.1) -> None:
        """Crude Newton-Raphson applied to log-likelihood. Updates self.beta."""

        minibatch_size = np.ceil( np.shape(train_xs)[0] / minibatches)
        print(f"Usual minibatch size: {minibatch_size}")
        train_zipped = np.column_stack((train_xs, train_ys))
        minibatches = [train_zipped[n: n + minibatch_size] for n in range(0, train_zipped.shape[0], step=minibatch_size)]

        for _ in tqdm.trange(epochs, desc="log-likelihood fit"):
            for batch in minibatches:
                xs_sample = batch[:, :-1]
                ys_sample = batch[:,-1]
                grad = self._log_likelihood_gradient(xs_sample, ys_sample, self.beta)
                hess = self._log_likelihood_hessian(xs_sample, self.beta)
                self.beta -= step_size * np.linalg.inv(hess) * grad
                



#
# Simple Linear Regression
#

# Correlation will measure whether a linear relationship exists between two variables
#   This is not enough
# Using a linear regression we can predict the exact nature of the relationship
#   Warning: this is assuming such a relationship exists!

# Hypothesis: 
#   Let x, y be the two variables
#   There exist beta = beta_0, beta_1 so that
#       y_i = beta_1 * x_i + beta_0 + e_i     for all i
#   where e_i is some error term

# Consider x to be the inputs and y to be the outputs

# Assuming we have `beta_0` and `beta_1`, we can predict outputs from new inputs:
def predict(beta_0: float, beta_1: float, x_i: float) -> float:
    return beta_1 * x_i + beta_0

# To choose `beta_0` and `beta_1`, we will want to minimize error
#   We will actually minimize the sum of squared errors from the dataset
#       squared so that positive and negative error dont cancel out

def error(beta_0: float, beta_1: float, x_i: float, y_i: float) -> float:
    """The error of predicting beta_1 * x_i + beta_0 
       when predicting y_i"""
    return predict(beta_0, beta_1, x_i) - y_i

def least_squares_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float]:
    """returns beta_0, beta_1 such that y=beta_1x+beta_0 has least squared error"""
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    std_y = np.std(ys)
    std_x = np.std(xs)
    corr_xy = np.corrcoef(xs, ys)[0][1]

    # These are, in fact, the optimal values
    #   Can solve with calculus
    b = corr_xy * std_y / std_x         # kind of like rise/run with corr giving the strength (and sign) 
                                        #   of this interpretation
    beta_0 = mean_y - beta_1 * mean_x         # if x == mean_x, 
                                        #   then y == beta_1 * x + beta_0
                                        #          == beta_1 * mean_x + mean_y - beta_1 * mean_x
                                        #          == mean_y
    return beta_0, beta_1

# Test
x = [i for i in range(-100, 110, 10)]
y = [2 * i - 5 for i in x]

print(least_squares_fit(x, y))  # Should be (-5, 3)

# How well does the optimized model fit the data?
#   Calculate 'R-squared' AKA 'coefficient of determination'

# Unexplained variation: error in model
def sum_of_sq_errors(beta_0:float, beta_1: float, xs: np.ndarray, ys: np.ndarray) -> float:
    return np.sum(np.array([error(beta_0, beta_1, x_i, y_i) **2  
                            for x_i, y_i in zip(xs, ys)]))

# 
def total_sum_of_squares(y: np.ndarray) -> float:
    """the total squared variation of y_i's from their mean"""
    return np.sum(np.square(y - np.linalg.norm(y)))

#
def r_squared(beta_0: float, beta_1: float, x: np.ndarray, y: np.ndarray) -> float:
    """the fraction of variation in y captured by the model, which equals
       1 - unexplained variation in y / total variation in y"""
       return 1.0 - (sum_of_sq_errors(beta_0, beta_1, x, y) / total_sum_of_squares(y))


#
# Note: 
#   We can also solve for the model using gradient descent
#       Probably shouldn't do this in practice, 
#       but we can try, for practice :)

from GradientDescent import gradient_step
import random
import tqdm

# Want to pick (beta_0, beta_1) that minimize mean squared error 
#   sum( e^2 ) = sum( (y_i - beta_1 * x_i - beta_0) ** 2 for i in range(len(xs)) )
#              = sum( y_i**2 + (beta_1*x_i)**2 + beta_0**2 - y_i*beta_1*x_i - y_i*beta_0 + beta_1*x_i*beta_0 )
# Compute the gradient explicitly:
#   grad = ( sum( 2 * beta_0 - y_i + beta_1 * x_i ),
#            sum( (2 * x_i**2) * beta_1 - y_i * x_i ) )
#        = (-2 * sum( error( beta_0, beta_1, x_i, y_i ) ),
#           -2 * sum( x_i * error( beta_0, beta_1, x_i, y_i ) ))      # supposed to be negative?


#   xs = num_friends_good
#   ys = daily_minutes_good
#
#   num epochs = 10000
#   learning_rate = 0.00001
#   guess = np.ndarray([random.random(), random.random()])
#   
#   with tqdm.trange(num_epochs) as t:
#       for _ in t:
#           beta_0, beta_1 = guess
#           
#           grad_0 = -2 * sum( error(beta_0, beta_1, x_i, y_i) for x_i, y_i in 
#                              for x_i, y_i in zip(xs, ys) )
#           grad_1 = -2 * sum( x_i * error(beta_0, beta_1, x_i, y_i) 
#                              for x_i, y_i in zip(xs, ys) )
#           grad = np.ndarray([grad_0, grad_1])

#           # tqdm
#           loss = sum_of_sq_errors(beta_0, beta_1, xs, ys)
#           t.set_description(f"loss: {loss:.3f}")
#
#           guess = gradient_step(guess, grad, step_size = -learning_rate])
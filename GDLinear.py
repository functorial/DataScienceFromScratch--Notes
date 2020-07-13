import numpy as np
import random

# x ranges from -50 to 49, y = 20x+5
input = [(x, 20*x+5) for x in range(-50, 50)]

# pretend we dont know that y = 20x+5
# we will use gradient descent to discover this fact

# start with a function that determines the gradient based on error from a single data point

# `theta` is supposed to be a guess about the slope and y-intercept of a linear model
# (x,y) is a data point
def linear_gradient(x: float, y: float, theta: np.ndarray) -> np.ndarray:
    slope, intercept = theta
    predicted = slope * x + intercept               # the prediction of the model
    error = (predicted - y)                         # error is (predicted - actual)
    squared_error = error ** 2                      # we will minimize squared error (error^2 is non-negative)
    grad = np.array([2 * error * x, 2 * error])     # using its gradient (error considered as a function of `slope` and `intercept`)
                                                    #
                                                    # move in direction of negative grad to minimize error
                                                    # assume error is positive, i.e. our guess is too large
                                                    # if x>0 (x<0), then increasing (decreasing) slope will decrease error
                                                    # increasing y-intercept will decrease error no matter what
    return grad

# Instead of having `error` track only one point,
# we can use mean-squared error to get a measure of error on the whole set
# The gradient of mean squared error is the mean of the individual gradients

#We will need this
def gradient_step(v: np.ndarray, gradient: np.ndarray,  step_size: float) -> np.ndarray:
    """Moves `step_side` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = step_size * gradient
    return v + step

# 1) Start with a random value for `theta`
# 2) Compute the mean of the gradients
# 3) Adjust `theta` in that direction
# 4) repeat 5000 times

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001       # step rate

for epoch in range(5000):
    grad = sum(np.array([linear_gradient(x, y, theta) for x, y, in input])) / len(input)
    theta = gradient_step(theta, grad, -learning_rate)
slope, intercept = theta
print(theta)                # ~[20,5]
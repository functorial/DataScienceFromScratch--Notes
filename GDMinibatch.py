from typing import TypeVar, List, Iterator
import numpy as np
import random
T = TypeVar('T')    # this allows us to type "generic" functions

input = [(x, 20*x+5) for x in range(-50, 50)]
def gradient_step(v: np.ndarray, gradient: np.ndarray,  step_size: float) -> np.ndarray:
    """Moves `step_side` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = step_size * gradient
    return v + step
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

def minibatches(dataset: List[T], batch_size: int, shuffle: bool=True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # start indices: 0, batch_size, 2*batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)] 

    if shuffle:
        random.shuffle(batch_starts)    # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]        # will return different batches every call

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(input, batch_size=20):
        grad = sum([linear_gradient(x, y, theta) for x, y in batch]) / len(batch)
        theta = gradient_step(theta, grad, -0.001)
    print(epoch, theta)

# The book also outlines a "stochastic gradient descent"
# which is the special case when `batch_size = 1`
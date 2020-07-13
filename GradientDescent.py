import numpy as np
import random

def gradient_step(v: np.ndarray, gradient: np.ndarray,  step_size: float) -> np.ndarray:
    """Moves `step_side` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = step_size * gradient
    return v + step

def sum_of_squares_gradient(v: np.ndarray) -> np.ndarray:
    #gradient of f(x,y,z)=x^2+y^2+z^2 is (2x, 2y, 2z)
    return 2 * v

# picks a random starting point in R^3
v = np.array([random.uniform(-10,10) for i in range(3)])

# take 1000 steps
# for epoch in range(1000):
#     grad = sum_of_squares_gradient(v)   # compute gradient at v
#     v = gradient_step(v, grad, -0.01)   # take a negative gradient step
#     print(epoch, v)

# print(np.linalg.norm(v))                # v should be close to 0
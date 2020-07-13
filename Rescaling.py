#
# Rescaling
#

# Many techniques are sensitive to the scale of your data

# Suppose we want to identify clusters of body sizes

# Person    Height (in)     Height (cm)     Weight (lbs)
# A         63              160             150
# B         67              170.2           160
# C         70              177.8           171

# Measure height in inches and find nearest neighbor via the (height, weight) pair
#   find that B's nearest neighbor is A

# Measure height in centimeters and find nearest neighbor via the (height, weight) pair
#   find that B's nearest neighbor is C

# PROBLEM! answer shouldn't depend on units
#   Rescale the data so that each dimension has MEAN=0, STD_DEV=1
#   Then the data will be unit-less

import numpy as np
from typing import List

# First, get the scales (means, stdevs)
def scale(data: List[float]) -> np.ndarray:
    """returns the mean and standard deviation for each position. Ignores case when stdev=0"""

    data_array = np.concatenate([np.array([list(v)]).T for v in data], axis=1)
    means = np.mean(data_array, axis=1)
    # ddof is "delta degrees of freedom", used in divisor N-ddof
    # ddof=1 in sample mean, ddof=0 in population mean
    stdevs = np.std(data_array, axis=1, ddof=1)

    return means, stdevs

# test
vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
print(f"Means: \t\t\t{means}")
print(f"Standard Deviations: \t{stdevs}")

# Use these to rescale original data
def rescale(data: List[float]) -> np.ndarray:
    """Rescales the input data so that each position has mean 0 and stdev 1"""
    means, stdevs = scale(data)
    dim = len(data[0])

    # slick way to make a copy of each vector in a list
    rescaled = np.array([v[:] for v in data])

    for v in rescaled:
        for i in range(dim):
            # ignore if stdev = 0
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    
    return rescaled

# test
means, stdevs = scale(rescale(vectors))
print("Rescale...")
print(f"New means: {means}")
print(f"New Stdevs: {stdevs}")
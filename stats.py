import math
import matplotlib.pyplot as plt
import random
from collections import Counter
from typing import Tuple, List


#
# Chaper 5
#

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

# The underscores indicate that these are "private" functions, as they're 
# not intended to be called by other people useing the stats library
def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, the median is the average of the two middle points"""
    sorted_xs = sorted(xs)
    hi_mdpt = len(xs) // 2
    return (sorted_xs[hi_mdpt] + sorted_xs[hi_mdpt-1]) / 2

def median(v: List[float]) -> float:
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]

def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [p for p, count in counts.items() if count == max_count]

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

def de_mean(xs: List[float]) -> List[float]:
    """Translate xs be subtracting its mean (so the result has mean 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at lease two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum([x ** 2 for x in deviations]) / (n - 1)

def standard_deviation(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

def interquartile_range(xs: List[float]) -> float:
    """Returns the difference between the 75%-ile and the 25%-ile"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have the same length"

    n = len(xs)
    z = zip(de_mean(xs), de_mean(ys))
    return sum([x*y for x, y in z]) / (n - 1)

# careful for Simpson's Paradox
def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0



#
# Chapter 6
#


sqrt_two_pi = math.sqrt(2 * math.pi)
def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    return math.exp(-(x-mu)**2 / 2 / sigma**2) / sqrt_two_pi / sigma

def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) /2

def inverse_normal_cdf(p: float, mu: float=0, sigma: float=1, tolerance: float=0.00001) -> float:
    """Find approximate inverse using binary search. This is possible since normal_cdf is continuous and monotonically increasing."""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10                     # normal_cdf(-10) is very close to 0
    hi_z = 10                       # normal_cdf(10) is very close to 1

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2      # Consider the midpoint
        mid_p = normal_cdf(mid_z)       # and the CDF's value there
        if mid_p < p:
            low_z = mid_z               # Midpoint too low, search above it
        else:
            hi_z = mid_z                # Midpoint too high, search below it
    return mid_z

def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    return sum(bernoulli_trial(p) for _ in range(n))

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a binomial(n,p) and plots their histogram."""
    data = [binomial(n,p) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    
    mu = p * n
    sigma = math.sqrt(n * p * (1-p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()



#
# Chapter 7
#


# Used for testing hypotheses about binomially distributed variables.
def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1-p) * n)
    return mu, sigma

# The normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float, mu: float=0, sigma: float=1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float=1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo: float, hi: float, mu: float, sigma: float) -> float:
    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability: float, mu: float=0, sigma: float=1) -> float:
    """Returns the z for which P(Z <= z) = probility"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float=0, sigma: float=1) -> float:
    """Returns the z for which P(Z >= z) = probility"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float, mu: float=0, sigma: float=1) -> Tuple[float, float]:
    """Returns the symmetric (about the mean) bounds which contain the specified probability"""
    tail_probability = (1 - probability) /2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound


# Experiment: Given a coin, find how fair it is. 
#
# Hypothesize that a given coin is fair. In other words, P(Heads) = 0.5
#
# Test: We can check this hypothesis by sampling it 1000 times and checking how it measures against its associated normal approximation
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)     # Assuming the coin is fair, this test will give the correct result 95% of the time
#
# W




def two_sided_p_value(x: float, mu: float=0, sigma: float=1) -> float:
    """How likely are we to see a value at least as rare as x (in either direction) if our values are from an N(mu, sigma)?"""
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        # using the fact that N(mu, sigma) is not skewed
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)

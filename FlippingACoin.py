from typing import Tuple
from stats import normal_cdf, inverse_normal_cdf
import math

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
    lower_bound = normal_upper_bound(tail_probability, mu, signma)

    return lower_bound, upper_bound

# Hypothesize that aa given coin is fair.
# Let's check hypothesis by sampling it 1000 times and checking how it measures against the corresponding normal approximation
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
# Assuming the coin is fair, this test will give the correct result 95% of the time

def two_sided_p_value(x: float, mu: float=0, sigma: float=1) -> float:
    """How likely are we to see a vlue at least as rare as x (in either direction) if our values are from an N(mu, sigma)?"""
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        # using the fact that N(mu, sigma) is not skewed
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)



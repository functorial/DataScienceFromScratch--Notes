import stats
from collections import Counter
import matplotlib.pyplot as plt
from math import sqrt

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n,p) and plots their histogram."""
    data = [stats.binomial(n,p) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    
    mu = p * n
    sigma = sqrt(n * p * (1-p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [stats.normal_cdf(i + 0.5, mu, sigma) - stats.normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()

binomial_histogram(0.50, 100, 10000)




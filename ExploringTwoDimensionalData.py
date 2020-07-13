from stats import inverse_normal_cdf
import matplotlib.pyplot as plt
import random
from math import floor
from collections import Counter


def random_normal() -> float:
    """Returns a radom draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

# individually, each dimension has a similar histogram
bucket_size = 0.1
buckets1 = [bucket_size * (floor(p / bucket_size)) for p in ys1]
buckets2 = [bucket_size * (floor(p / bucket_size)) for p in ys2]
h1 = Counter(buckets1)
h2 = Counter(buckets2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.bar(h1.keys(), h1.values(), width=bucket_size)
ax1.set_xlabel('ys1')
ax1.set_ylabel('counts')
ax1.set_title('similar to ys2 counts...')

ax2.bar(h2.keys(), h2.values(), width=bucket_size)
ax2.set_xlabel('ys2')
ax2.set_title('similar to ys1 counts')

plt.tight_layout()
plt.show()


# but each has a very different joint distribution with xs
plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions With xs")
plt.show()

from stats import correlation
print(correlation(xs, ys1))
print(correlation(xs, ys2))
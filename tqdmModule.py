#
# tqdm
#

# This is a way to see how long your computation has to go
# Can create progress bars

import tqdm
import random

# An iterable wrapped in tqdm.tqdm will produce a progress bar
for i in tqdm.tqdm(range(100)):
    #something slow
    _ = [random.random() for _ in range(100000)]

# Here's a shorthand for the `range` wrapper
for i in tqdm.trange(100):
    #something slow
    _ = [random.random() for _ in range(100000)]

# Can also set description of progress bar
# Use a with statement:
def primes(n):
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i in t:
            i_is_prime = not any(i%p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)

            t.set_description(f"{len(primes)} primes")

    return primes

# _ has no meaning, allows easier reading for large numbers
my_primes = primes(50_000)
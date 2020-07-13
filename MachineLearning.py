#
# Meachine Learning
#

# Machine learning is creating and using models that are learned from data
#   Supervised Models:
#       Using data with correct answers to learn from
#   Unsupervised Models:
#       No such labels
#   Other types....

# Typically, we will have a parameterized family of models and
# we will learn the parameters that are optimal in some way.

# Danger:
#   We want to avoid overfitting our model to the data

#   We can try splitting out data set first
import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')    # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1-prob]"""
    data = data[:]                  # Make a shallow copy
    random.shuffle(data)            # because shuffle modifies the list
    cut = int(len(data) * prob)     # use prob to find cutoff
    return data[:cut], data[cut:]   # split the shuffled list there

# test
data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

assert len(train) == 750
assert len(test) == 250

# Often, we might have paired input variables and output variables
Y = TypeVar('Y')    #generic type to represent output variables

def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    assert len(xs) == len(ys)
    # generate indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])

# test
xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))

# If the model was overfit to training data, then it will
# hopefully perform poorly on the separate test data


#
# Correctness
#

# Don't use raw accuracy to determine how good a model is!!!!!!!!!!!!!!!!!!!
# This would be falling for the classic 'Bayes Theorem trap'

# Would be better to use other measures:
def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    """Returns accuracy of our positive predictions"""
    return tp / (tp + fp)

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    """Returns accuracy for total positives"""
    return tp / (tp + fn)

# Sometimes these two measures are combined into the 'F1-score'
#   This is the harmonic mean of the two!
#       Harmonic mean is appropriate for situations when the average of rates is desired
def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)

